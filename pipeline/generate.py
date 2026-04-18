"""
End-to-end pipeline: (текстовый промт + сегментный план) → полный MIDI-трек.

Шаги:
  1. Text2Prefix : text prompt  → piano roll prefix [T_prefix, 128]
  2. AffinitySSM : segment_plan → SSM [T_total, T_total]
  3. SingLS      : prefix + SSM → full piano roll   [T_total, 128]
  4. Export      : MIDI + visualizations

Использование:
  python -m pipeline.generate \\
      --model models/checkpoints/your_model.pt \\
      --prompt "gentle piano melody in C major" \\
      --segment "intro:8,verse:16,chorus:16,verse:16,chorus:16,outro:8" \\
      --tempo 90 \\
      --n_prefix_bars 8 \\
      --n_gen_bars 64 \\
      --ssm_type affinity \\
      --out_dir outputs/run1
"""

import argparse
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch

from SingLS.config.config import DEVICE
from SingLS.trainer.data_utils import topk_batch_sample
from Seg2SSM.affinity_ssm import AffinitySSM, LABEL_NAMES

# Предпочитаем прямой MIDI-генератор (нет шума транскрипции).
# Fallback на MusicGen+BasicPitch, если text2midi не установлен.
try:
    from Text2Prefix.text2prefix_midi import Text2PrefixMIDI as Text2Prefix
    _PREFIX_BACKEND = "text2midi"
except ImportError:
    from Text2Prefix.text2prefix import Text2Prefix          # type: ignore[assignment]
    _PREFIX_BACKEND = "musicgen+basicpitch"


# ── Константы ──────────────────────────────────────────────────────────────

BEATS_PER_BAR      = 4
_DEFAULT_PREFIX_BARS = 8   # бюджет токенов для text2midi; реальный T_prefix определяется по shape

LABEL_NAME_TO_ID = {v: k for k, v in LABEL_NAMES.items()}


# ── Тип SSM для ablation ────────────────────────────────────────────────────

class SSMType(Enum):
    AFFINITY  = "affinity"   # Теоретическая матрица A_fixed
    EMPIRICAL = "empirical"  # Эмпирическая матрица A из SALAMI
    RANDOM    = "random"     # Случайная симметричная SSM (нижняя граница)
    NONE      = "none"       # Нулевая SSM (модель игнорирует структуру)


# ── Парсинг сегментного плана ───────────────────────────────────────────────

def parse_segment_plan(plan_str: str) -> List[Tuple[int, int]]:
    """
    Парсит сегментный план из строки формата "intro:8,verse:16,chorus:16".

    Returns:
        [(label_id, n_bars), ...]
    """
    plan = []
    for token in plan_str.split(","):
        token = token.strip()
        if ":" not in token:
            raise ValueError(
                f"Неверный формат токена: '{token}'. Ожидается 'label:n_bars'."
            )
        name, bars = token.rsplit(":", 1)
        name = name.strip().lower()
        if name not in LABEL_NAME_TO_ID:
            raise ValueError(
                f"Неизвестная метка секции: '{name}'. "
                f"Доступные: {sorted(LABEL_NAME_TO_ID.keys())}"
            )
        plan.append((LABEL_NAME_TO_ID[name], int(bars)))
    return plan


# ── Построение SSM ──────────────────────────────────────────────────────────

def build_ssm(
    ssm_type: SSMType,
    segment_plan: List[Tuple[int, int]],
    total_len: int,
    empirical_ckpt: Optional[str] = None,
) -> torch.Tensor:
    """
    Строит SSM [total_len, total_len] по выбранному режиму.

    Args:
        ssm_type      : тип SSM (affinity / empirical / random / none)
        segment_plan  : [(label_id, n_bars), ...]
        total_len     : T_prefix + T_gen (количество битов)
        empirical_ckpt: путь к чекпоинту A_empirical (нужен для empirical)
    """
    if ssm_type == SSMType.AFFINITY:
        return AffinitySSM.fixed().build(segment_plan, ssm_size=total_len)

    if ssm_type == SSMType.EMPIRICAL:
        if empirical_ckpt is None:
            raise ValueError("--empirical_ckpt обязателен при ssm_type=empirical")
        return AffinitySSM.from_checkpoint(empirical_ckpt).build(
            segment_plan, ssm_size=total_len
        )

    if ssm_type == SSMType.RANDOM:
        rand = torch.rand(total_len, total_len)
        rand = (rand + rand.T) / 2   # симметричная
        rand.fill_diagonal_(1.0)
        return rand

    if ssm_type == SSMType.NONE:
        # Нулевая матрица: модели с ORIGINAL/LSA/LSA_SB вниманием получат
        # нулевые веса, что близко к базовому NONE-режиму.
        return torch.zeros(total_len, total_len)

    raise ValueError(f"Unknown SSMType: {ssm_type}")


# ── Авторегрессивная генерация ──────────────────────────────────────────────

def generate_from_prefix(
    model,
    prefix: torch.Tensor,       # [T_prefix, 128]
    batched_ssm: torch.Tensor,  # [T_total, T_total]
    gen_len: int,
    warmup_beats: int = 4,
    temperature: float = 1.5,
    prefix_in_sequence: bool = False,  # True: OOD-фреймы видны attention; False: Fix 2 из COLLAPSE_DEBUG
) -> torch.Tensor:
    """
    Авторегрессивная генерация с заданным prefix и SSM.

    Проблема: если пропустить весь OOD-prefix через LSTM, hidden state
    уходит в патологическую область → генерация коллапсирует к 2–3 нотам.

    Решение:
      - init_hidden (нули) вместо set_random_hidden (случайный шум)
      - через LSTM пускаем только последние warmup_beats битов prefix
      - prefix_in_sequence=False (Fix 2): OOD-фреймы НЕ добавляются в sequence,
        attention видит только сгенерированные in-distribution ноты
      - prefix_in_sequence=True: старое поведение (prefix в sequence, OOD виден)

    Returns:
        sequence [T_prefix + gen_steps, 1, 128]  (если prefix_in_sequence=True)
                 [gen_steps, 1, 128]              (если prefix_in_sequence=False)
    """
    model.eval()
    model.init_hidden(1)   # нулевая инициализация, не random

    prefix_t = prefix.unsqueeze(1).to(DEVICE)   # [T_prefix, 1, 128]

    # Warmup: через LSTM пропускаем только последние warmup_beats битов prefix,
    # чтобы прогреть hidden state без патологического накопления OOD-ошибок.
    # Важно: prev_sequence для warmup должен быть непустым (sparsemax падает на
    # пустом тензоре). Используем полный prefix_t — нас интересует только hidden state.
    warmup = min(warmup_beats, prefix_t.shape[0])
    if warmup > 0:
        warmup_input = prefix_t[-warmup:]       # [warmup, 1, 128]
        with torch.no_grad():
            model.forward(warmup_input.float(), 1, prefix_t, batched_ssm)

    # Начальная sequence для авторегрессии.
    # Важно: sequence не может быть пустой — sparsemax в original_attention
    # упадёт на пустом срезе SSM (IndexError: max() on empty tensor).
    # При prefix_in_sequence=False берём только последний фрейм prefix —
    # минимальное OOD-загрязнение, но attention получает хотя бы 1 фрейм.
    if prefix_in_sequence:
        sequence = prefix_t                     # [T_prefix, 1, 128]
    else:
        sequence = prefix_t[-1:]                # [1, 1, 128] — один последний фрейм

    max_notes = batched_ssm.shape[0] - prefix_t.shape[0]
    steps = min(gen_len, max_notes)

    # Первый генерируемый шаг: авторегрессия с last warmup frame как входом
    if warmup > 0:
        next_element = prefix_t[-1:]            # [1, 1, 128] — последний бит prefix
    else:
        next_element = torch.zeros(1, 1, 128, device=DEVICE)

    for _ in range(steps):
        with torch.no_grad():
            output, _ = model.forward(
                next_element.float(), 1, sequence, batched_ssm
            )
            next_element = topk_batch_sample(output, 50, temperature=temperature)  # [1, 1, 128]
        sequence = torch.vstack((sequence, next_element.to(DEVICE)))

    # Собираем полный sequence для визуализации.
    # При prefix_in_sequence=False: sequence = [last_prefix_frame] + [generated].
    # Добавляем оставшиеся prefix-фреймы спереди для корректного отображения.
    if prefix_in_sequence:
        full_sequence = sequence
    else:
        full_sequence = torch.vstack((prefix_t[:-1], sequence))  # [T_prefix + steps, 1, 128]
    return full_sequence


# ── MIDI экспорт ────────────────────────────────────────────────────────────

def piano_roll_to_midi(
    piano_roll: np.ndarray,   # [T, 128]
    tempo: float,
) -> pretty_midi.PrettyMIDI:
    """Конвертирует piano roll в pretty_midi.PrettyMIDI."""
    fs = tempo / 60.0
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    pr = piano_roll.T  # [128, T]
    for note in range(pr.shape[0]):
        on = None
        for t in range(pr.shape[1]):
            if pr[note, t] > 0 and on is None:
                on = t
            elif pr[note, t] == 0 and on is not None:
                instrument.notes.append(pretty_midi.Note(
                    velocity=90, pitch=note,
                    start=on / fs, end=t / fs,
                ))
                on = None
        if on is not None:
            instrument.notes.append(pretty_midi.Note(
                velocity=90, pitch=note,
                start=on / fs, end=pr.shape[1] / fs,
            ))

    midi.instruments.append(instrument)
    return midi


# ── Визуализация ────────────────────────────────────────────────────────────

def save_visualizations(
    out_dir: Path,
    prefix_roll: torch.Tensor,   # [T_prefix, 128]
    full_roll: np.ndarray,       # [T_total, 128]
    ssm: torch.Tensor,           # [T_total, T_total]
    tempo: float,
    ssm_type: str,
    segment_plan: List[Tuple[int, int]],
):
    T_prefix = prefix_roll.shape[0]
    T_total = full_roll.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Prefix piano roll
    axes[0].imshow(prefix_roll.numpy().T, aspect="auto", origin="lower",
                   cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title(f"Prefix (Text2Prefix)\nT={T_prefix} beats, tempo={tempo:.0f} BPM")
    axes[0].set_xlabel("Beat")
    axes[0].set_ylabel("MIDI note")

    # 2. Full piano roll с границей prefix
    axes[1].imshow(full_roll.T, aspect="auto", origin="lower",
                   cmap="Blues", vmin=0, vmax=1)
    axes[1].axvline(T_prefix - 0.5, color="red", lw=1.5,
                    linestyle="--", label=f"prefix end (beat {T_prefix})")
    axes[1].set_title(f"Full Track (SingLS)\nT={T_total} beats")
    axes[1].set_xlabel("Beat")
    axes[1].legend(fontsize=8, loc="upper right")

    # 3. SSM heatmap с сегментными границами
    ssm_np = ssm.numpy()
    im = axes[2].imshow(ssm_np, aspect="auto", origin="lower",
                        cmap="hot", vmin=0, vmax=1)
    axes[2].axvline(T_prefix - 0.5, color="cyan", lw=1.0, linestyle="--")
    axes[2].axhline(T_prefix - 0.5, color="cyan", lw=1.0, linestyle="--")
    # Границы секций (в битах)
    cursor = 0
    for _, n_bars in segment_plan:
        cursor += n_bars * BEATS_PER_BAR
        if cursor < T_total:
            axes[2].axvline(cursor - 0.5, color="white", lw=0.5, alpha=0.6)
            axes[2].axhline(cursor - 0.5, color="white", lw=0.5, alpha=0.6)
    axes[2].set_title(f"SSM [{ssm_type}]\n{T_total}×{T_total}")
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    # Подпись сегментного плана
    plan_str = " → ".join(
        f"{LABEL_NAMES.get(lbl, '?')}:{bars}b"
        for lbl, bars in segment_plan
    )
    plt.suptitle(f"End-to-end pipeline | {plan_str}", fontsize=11)
    plt.tight_layout()

    path = out_dir / "pipeline_overview.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved overview : {path}")


# ── Главная функция пайплайна ────────────────────────────────────────────────

def run_pipeline(
    model_path: str,
    prompt: str,
    segment_plan: List[Tuple[int, int]],
    tempo: float = 120.0,
    n_gen_bars: int = 64,
    ssm_type: SSMType = SSMType.AFFINITY,
    empirical_ckpt: Optional[str] = None,
    out_dir: str = "outputs",
    warmup_beats: int = 4,
    temperature: float = 1.5,
    prefix_pt: Optional[str] = None,
    prefix_in_sequence: bool = False,  # False = Fix 2: OOD prefix не виден attention
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Запускает полный end-to-end пайплайн.

    Returns:
        full_roll    : np.ndarray [T_total, 128]
        prefix_roll  : torch.Tensor [T_prefix, 128]
        ssm          : torch.Tensor [T_total, T_total]
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    T_prefix_requested = _DEFAULT_PREFIX_BARS * BEATS_PER_BAR
    T_gen    = n_gen_bars * BEATS_PER_BAR

    print("\n" + "=" * 60)
    print("End-to-End Music Generation Pipeline")
    print("=" * 60)
    print(f"  Prompt        : {prompt}")
    plan_str = ", ".join(
        f"{LABEL_NAMES.get(l, '?')}:{b}b" for l, b in segment_plan
    )
    print(f"  Segment plan  : {plan_str}")
    print(f"  Tempo         : {tempo} BPM")
    print(f"  Prefix budget : {_DEFAULT_PREFIX_BARS} bars max ({T_prefix_requested} beats)")
    print(f"  Generation    : {n_gen_bars} bars ({T_gen} beats)")
    print(f"  SSM type      : {ssm_type.value}")
    print(f"  Warmup beats  : {warmup_beats}")
    print(f"  Prefix in seq : {prefix_in_sequence}  ({'OOD visible to attention' if prefix_in_sequence else 'Fix2: clean attention context'})")
    print(f"  Output dir    : {out}")

    # ── Step 1: Text → Prefix ──────────────────────────────────────────────
    if prefix_pt is not None:
        print(f"\n[1/4] Loading saved prefix from {prefix_pt}...")
        saved = torch.load(prefix_pt, weights_only=False)
        prefix_roll   = saved["prefix_roll"]    # [T_prefix, 128]
        detected_tempo = saved["tempo"]
        print(f"  prefix shape : {tuple(prefix_roll.shape)}")
        print(f"  tempo        : {detected_tempo:.1f} BPM  (loaded)")
    else:
        print(f"\n[1/4] Text → Prefix  (backend: {_PREFIX_BACKEND})...")
        text2prefix = Text2Prefix()
        prefix_roll, detected_tempo, num_beats = text2prefix.generate(
            prompt=prompt,
            n_bars=_DEFAULT_PREFIX_BARS,
            tempo=tempo,
        )
        print(f"  prefix shape : {tuple(prefix_roll.shape)}")
        print(f"  tempo        : {detected_tempo:.1f} BPM")

    # Пересчитываем T_prefix из реальной формы prefix_roll.
    # text2midi может завершиться раньше (EOS) → prefix_roll короче запрошенного.
    # T_prefix = prefix_roll.shape[0] соответствует реальному контенту (после тримминга).
    T_prefix = prefix_roll.shape[0]
    T_total  = T_prefix + T_gen
    if T_prefix != T_prefix_requested:
        print(
            f"  [!] Prefix trimmed: {T_prefix_requested} → {T_prefix} beats "
            f"({T_prefix_requested // BEATS_PER_BAR} → {T_prefix // BEATS_PER_BAR} bars). "
            f"Красная линия сдвинута влево."
        )

    # Сохраняем prefix чтобы можно было переиспользовать в других прогонах
    prefix_save_path = out / "prefix_roll.pt"
    torch.save({"prefix_roll": prefix_roll, "tempo": detected_tempo}, prefix_save_path)
    print(f"  Saved prefix   : {prefix_save_path}")

    # ── Step 2: Segment plan → SSM ─────────────────────────────────────────
    print("\n[2/4] Segment plan → SSM (AffinitySSM)...")
    ssm = build_ssm(ssm_type, segment_plan, T_total, empirical_ckpt)
    ssm = ssm.to(DEVICE)
    print(f"  SSM shape : {tuple(ssm.shape)}")
    print(f"  SSM stats : mean={ssm.mean():.3f}, min={ssm.min():.3f}, max={ssm.max():.3f}")

    # ── Step 3: SingLS generation ──────────────────────────────────────────
    print("\n[3/4] Loading model and generating...")
    model = torch.load(model_path, weights_only=False, map_location=DEVICE)
    model.eval()
    print(f"  Model type : {type(model).__name__}")
    print(f"  Generating {T_gen} beats ({n_gen_bars} bars)...")

    sequence = generate_from_prefix(
        model, prefix_roll, ssm, T_gen,
        warmup_beats, temperature,
        prefix_in_sequence=prefix_in_sequence,
    )
    full_roll = sequence.squeeze(1).detach().cpu().numpy().round()
    full_roll = full_roll[:T_total]
    print(f"  Output shape : {full_roll.shape}")

    # ── Step 4: Export ─────────────────────────────────────────────────────
    print("\n[4/4] Exporting results...")

    midi = piano_roll_to_midi(full_roll, detected_tempo)
    midi_path = out / "generated.mid"
    midi.write(str(midi_path))
    print(f"  Saved MIDI         : {midi_path}")

    prefix_midi = piano_roll_to_midi(prefix_roll.numpy(), detected_tempo)
    prefix_midi.write(str(out / "prefix.mid"))
    print(f"  Saved prefix MIDI  : {out / 'prefix.mid'}")

    save_visualizations(
        out, prefix_roll, full_roll, ssm.cpu(),
        detected_tempo, ssm_type.value, segment_plan,
    )

    print(f"\n{'=' * 60}")
    print(f"Done. Results saved to {out}/")
    print("=" * 60)

    return full_roll, prefix_roll, ssm


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end music generation: text + segment plan → MIDI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Путь к чекпоинту SingLS (.pt), например:\n"
             "  models/checkpoints/trained_transformer_original_less_struct_combined.pt",
    )
    parser.add_argument(
        "--prompt", required=True,
        help='Текстовый промт для MusicGen, например:\n"gentle piano melody in C major"',
    )
    parser.add_argument(
        "--segment", required=True,
        help=(
            "Сегментный план в формате 'label:n_bars,...'\n"
            "  Доступные метки: intro, verse, chorus, bridge, instr, outro, other\n"
            "  Пример: \"intro:8,verse:16,chorus:16,verse:16,chorus:16,outro:8\""
        ),
    )
    parser.add_argument(
        "--tempo", type=float, default=120.0,
        help="Темп в BPM (по умолчанию: 120)",
    )
    parser.add_argument(
        "--n_gen_bars", type=int, default=64,
        help="Длина генерируемой части в барах (по умолчанию: 64)",
    )
    parser.add_argument(
        "--ssm_type", default="affinity",
        choices=["affinity", "empirical", "random", "none"],
        help=(
            "Тип SSM для структурного prior:\n"
            "  affinity  — теоретическая A_fixed (по умолчанию)\n"
            "  empirical — эмпирическая A из SALAMI (нужен --empirical_ckpt)\n"
            "  random    — случайная SSM (нижняя граница ablation)\n"
            "  none      — нулевая SSM (baseline без структуры)"
        ),
    )
    parser.add_argument(
        "--empirical_ckpt",
        default="Seg2SSM/checkpoints/affinity_matrix.pt",
        help="Путь к эмпирической матрице A (нужен при ssm_type=empirical)",
    )
    parser.add_argument(
        "--out_dir", default="outputs",
        help="Директория для сохранения результатов (по умолчанию: outputs/)",
    )
    parser.add_argument(
        "--warmup_beats", type=int, default=4,
        help=(
            "Сколько последних битов prefix пропускать через LSTM (по умолчанию: 4).\n"
            "Меньше = меньше OOD-воздействия на hidden state → более стабильная генерация.\n"
            "Полный prefix всё равно виден SSM attention-механизму."
        ),
    )
    parser.add_argument(
        "--prefix_pt", default=None,
        help=(
            "Путь к сохранённому prefix_roll.pt (из предыдущего прогона).\n"
            "Если задан — MusicGen не запускается, prefix берётся из файла.\n"
            "Полезно для чистого сравнения (например, разные температуры на одном prefix)."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5,
        help=(
            "Температура сэмплирования (по умолчанию: 1.5).\n"
            "T=1.0 — без температуры (коллапс к 1–2 нотам).\n"
            "T>1.0 — более равномерное распределение → разнообразная генерация."
        ),
    )

    parser.add_argument(
        "--prefix_in_sequence", action="store_true", default=False,
        help=(
            "Если задан — OOD-фреймы prefix попадают в sequence и видны attention.\n"
            "По умолчанию (False) — Fix 2: attention видит только in-distribution ноты,\n"
            "что предотвращает коллапс при OOD prefix от text2midi."
        ),
    )

    args = parser.parse_args()

    segment_plan = parse_segment_plan(args.segment)
    ssm_type = SSMType(args.ssm_type)

    run_pipeline(
        model_path=args.model,
        prompt=args.prompt,
        segment_plan=segment_plan,
        tempo=args.tempo,
        n_gen_bars=args.n_gen_bars,
        ssm_type=ssm_type,
        empirical_ckpt=args.empirical_ckpt,
        out_dir=args.out_dir,
        warmup_beats=args.warmup_beats,
        temperature=args.temperature,
        prefix_pt=args.prefix_pt,
        prefix_in_sequence=args.prefix_in_sequence,
    )


if __name__ == "__main__":
    main()
