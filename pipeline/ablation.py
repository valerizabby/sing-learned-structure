"""
Ablation study: сравнение всех типов SSM на одном примере.

Генерирует трек с одним и тем же prefix + моделью, но с разными SSM:
  1. none      — нулевая SSM (baseline без структуры)
  2. random    — случайная SSM (нижняя граница)
  3. affinity  — теоретическая A_fixed (целевой режим)
  4. empirical — эмпирическая A из SALAMI (если чекпоинт доступен)

Для каждого варианта сохраняет MIDI и вычисляет SSM-метрики
(IOU и MSE относительно affinity SSM как эталона структуры).

Использование:
  python -m pipeline.ablation \\
      --model models/checkpoints/your_model.pt \\
      --prompt "gentle piano melody in C major" \\
      --segment "intro:8,verse:16,chorus:16,verse:16,chorus:16,outro:8" \\
      --tempo 90 \\
      --out_dir outputs/ablation
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from SingLS.config.config import DEVICE
from Seg2SSM.affinity_ssm import LABEL_NAMES
from pipeline.generate import (
    SSMType,
    _DEFAULT_PREFIX_BARS,
    build_ssm,
    generate_from_prefix,
    parse_segment_plan,
    piano_roll_to_midi,
)
from Text2Prefix.text2prefix import Text2Prefix


BEATS_PER_BAR = 4
GT_DATA_PATH  = "data/combined/combined_test.pt"
GT_N_TRACKS   = 20   # сколько GT треков усреднять для референса


# ── SSM-метрики ─────────────────────────────────────────────────────────────

def load_gt_reference_ssm(T_total: int, n_tracks: int = GT_N_TRACKS) -> np.ndarray:
    """
    Строит эталонную SSM как среднее по n_tracks треков из combined_test.pt.
    Каждая SSM обрезается/ресайзится до T_total × T_total через билинейную интерполяцию.
    """
    import torch as _torch
    from scipy.ndimage import zoom

    data = _torch.load(GT_DATA_PATH, weights_only=False)
    accumulated = np.zeros((T_total, T_total), dtype=np.float32)
    count = 0

    for item in data[:n_tracks]:
        roll = item[0]  # tensor [T, 128]
        if isinstance(roll, _torch.Tensor):
            roll = roll.numpy()
        if roll.ndim != 2 or roll.shape[1] != 128:
            continue

        ssm = compute_ssm_from_roll(roll.astype(np.float32))  # [T_gt, T_gt]
        T_gt = ssm.shape[0]
        if T_gt < 4:
            continue

        # Ресайз до T_total × T_total
        factor = T_total / T_gt
        ssm_resized = zoom(ssm, factor, order=1)  # билинейная интерполяция
        # zoom может дать размер ±1 из-за округления
        ssm_resized = ssm_resized[:T_total, :T_total]
        if ssm_resized.shape != (T_total, T_total):
            pad = T_total - ssm_resized.shape[0]
            ssm_resized = np.pad(ssm_resized, ((0, pad), (0, pad)), mode="edge")

        accumulated += ssm_resized
        count += 1

    if count == 0:
        raise RuntimeError("Не удалось загрузить ни одного GT трека из combined_test.pt")

    gt_ssm = accumulated / count
    print(f"  GT reference SSM: mean={gt_ssm.mean():.3f}, std={gt_ssm.std():.3f}  ({count} tracks)")
    return gt_ssm


def compute_ssm_from_roll(roll: np.ndarray) -> np.ndarray:
    """
    Вычисляет chroma-SSM из piano roll [T, 128].

    Повторяет логику SingLS/trainer/data_utils.py::SSM().
    """
    T = roll.shape[0]
    chroma = np.zeros((T, 12), dtype=np.float32)
    for note in range(12):
        chroma[:, note] = roll[:, note::12].sum(axis=1)
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    chroma = chroma / (norms + 1e-8)
    return chroma @ chroma.T   # [T, T]


def iou_ssm(ssm1: np.ndarray, ssm2: np.ndarray, threshold: float = 0.5) -> float:
    b1 = ssm1 > threshold
    b2 = ssm2 > threshold
    intersection = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    return float(intersection / (union + 1e-8))


def mse_ssm(ssm1: np.ndarray, ssm2: np.ndarray) -> float:
    n = min(ssm1.shape[0], ssm2.shape[0])
    return float(np.mean((ssm1[:n, :n] - ssm2[:n, :n]) ** 2))


# ── Визуализация ablation ────────────────────────────────────────────────────

def save_ablation_plot(
    results: dict,                         # {ssm_type_str: {"roll": ..., "ssm": ...}}
    reference_ssm: np.ndarray,             # аффинная SSM как структурный эталон
    out_dir: Path,
    segment_plan: List[Tuple[int, int]],
    T_prefix: int,
    T_total: int,
):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))

    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (ssm_name, data) in enumerate(results.items()):
        roll = data["roll"]           # [T_total, 128]
        ssm  = data["ssm_target"]    # [T_total, T_total]

        # Верхний ряд: piano roll
        axes[0, col].imshow(roll.T, aspect="auto", origin="lower",
                            cmap="Blues", vmin=0, vmax=1)
        axes[0, col].axvline(T_prefix - 0.5, color="red", lw=1.2, linestyle="--")
        iou = data["iou"]
        mse = data["mse"]
        axes[0, col].set_title(
            f"[{ssm_name}]\nIOU={iou:.3f}  MSE={mse:.4f}", fontsize=10
        )
        axes[0, col].set_xlabel("Beat")
        if col == 0:
            axes[0, col].set_ylabel("MIDI note")

        # Нижний ряд: SSM для генерации
        ssm_np = ssm if isinstance(ssm, np.ndarray) else ssm.numpy()
        im = axes[1, col].imshow(ssm_np, aspect="auto", origin="lower",
                                 cmap="hot", vmin=0, vmax=1)
        axes[1, col].axvline(T_prefix - 0.5, color="cyan", lw=1.0, linestyle="--")
        axes[1, col].axhline(T_prefix - 0.5, color="cyan", lw=1.0, linestyle="--")
        # Границы секций
        cursor = 0
        for _, n_bars in segment_plan:
            cursor += n_bars * BEATS_PER_BAR
            if cursor < T_total:
                axes[1, col].axvline(cursor - 0.5, color="white", lw=0.5, alpha=0.5)
                axes[1, col].axhline(cursor - 0.5, color="white", lw=0.5, alpha=0.5)
        axes[1, col].set_title(f"Input SSM [{ssm_name}]", fontsize=10)
        axes[1, col].set_xlabel("Beat")
        plt.colorbar(im, ax=axes[1, col], shrink=0.8)

    plan_str = " → ".join(
        f"{LABEL_NAMES.get(l, '?')}:{b}b" for l, b in segment_plan
    )
    plt.suptitle(f"Ablation Study — SSM type comparison\n{plan_str}", fontsize=12)
    plt.tight_layout()
    path = out_dir / "ablation_overview.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved ablation overview : {path}")


def save_metrics_table(results: dict, out_dir: Path):
    """Сохраняет текстовую таблицу метрик."""
    lines = ["SSM type   | IOU (↑)  | MSE (↓)"]
    lines.append("-" * 36)
    for name, data in results.items():
        lines.append(f"{name:<10} | {data['iou']:.4f}   | {data['mse']:.5f}")
    table = "\n".join(lines)
    print("\n" + table)
    (out_dir / "metrics.txt").write_text(table + "\n")


# ── Главная функция ablation ─────────────────────────────────────────────────

def run_ablation(
    model_path: str,
    prompt: str,
    segment_plan: List[Tuple[int, int]],
    tempo: float = 120.0,
    n_gen_bars: int = 64,
    empirical_ckpt: Optional[str] = None,
    out_dir: str = "outputs/ablation",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    T_gen = n_gen_bars * BEATS_PER_BAR

    print("\n" + "=" * 60)
    print("Ablation Study — SSM type comparison")
    print("=" * 60)
    plan_str = ", ".join(f"{LABEL_NAMES.get(l,'?')}:{b}b" for l, b in segment_plan)
    print(f"  Segment plan   : {plan_str}")
    print(f"  Tempo          : {tempo} BPM")
    print(f"  Prefix budget  : {_DEFAULT_PREFIX_BARS} bars max")
    print(f"  Generation     : {n_gen_bars} bars ({T_gen} beats)")
    print(f"  Output dir     : {out}")

    # ── Step 1: Генерируем prefix один раз ────────────────────────────────
    print("\n[1/3] Generating shared prefix...")
    text2prefix = Text2Prefix()
    prefix_roll, detected_tempo, _ = text2prefix.generate(
        prompt=prompt, n_bars=_DEFAULT_PREFIX_BARS, tempo=tempo,
    )
    # Реальный T_prefix берётся из shape (text2midi мог сгенерировать короче бюджета)
    T_prefix = prefix_roll.shape[0]
    T_total  = T_prefix + T_gen
    print(f"  prefix shape : {tuple(prefix_roll.shape)}  ({T_prefix // BEATS_PER_BAR} bars), tempo={detected_tempo:.1f}")

    # ── Step 2: Загружаем модель ───────────────────────────────────────────
    print("\n[2/3] Loading model...")
    model = torch.load(model_path, weights_only=False, map_location=DEVICE)
    model.eval()
    print(f"  Model type : {type(model).__name__}")

    # Эталонная SSM — среднее по GT трекам из combined_test.pt.
    # Использование реальных треков как референса честнее чем affinity SSM:
    # метрика не зависит от того, какой тип SSM подаётся на вход модели.
    print("\n[ref] Building GT reference SSM...")
    reference_ssm = load_gt_reference_ssm(T_total)

    # ── Step 3: Генерация для каждого типа SSM ────────────────────────────
    print("\n[3/3] Running ablation conditions...")

    conditions = [
        ("none",     SSMType.NONE),
        ("random",   SSMType.RANDOM),
        ("affinity", SSMType.AFFINITY),
    ]
    if empirical_ckpt and Path(empirical_ckpt).exists():
        conditions.append(("empirical", SSMType.EMPIRICAL))

    results = {}

    for name, ssm_type in conditions:
        print(f"\n  --- {name} ---")
        cond_dir = out / name
        cond_dir.mkdir(exist_ok=True)

        ssm = build_ssm(ssm_type, segment_plan, T_total, empirical_ckpt).to(DEVICE)
        print(f"  SSM: mean={ssm.mean():.3f}")

        sequence = generate_from_prefix(model, prefix_roll, ssm, T_gen, warmup_beats=4)
        full_roll = sequence.squeeze(1).detach().cpu().numpy().round()[:T_total]

        # MIDI
        midi = piano_roll_to_midi(full_roll, detected_tempo)
        midi.write(str(cond_dir / "generated.mid"))
        print(f"  Saved: {cond_dir / 'generated.mid'}")

        # Метрики относительно GT SSM (среднее по combined_test)
        gen_ssm = compute_ssm_from_roll(full_roll)
        iou = iou_ssm(gen_ssm, reference_ssm)
        mse = mse_ssm(gen_ssm, reference_ssm)
        print(f"  IOU={iou:.4f}  MSE={mse:.5f}")

        results[name] = {
            "roll": full_roll,
            "ssm_target": ssm.cpu(),
            "gen_ssm": gen_ssm,
            "iou": iou,
            "mse": mse,
        }

    # ── Вывод и сохранение ─────────────────────────────────────────────────
    save_metrics_table(results, out)
    save_ablation_plot(results, reference_ssm, out, segment_plan, T_prefix, T_total)

    # Сохраняем prefix MIDI
    piano_roll_to_midi(prefix_roll.numpy(), detected_tempo).write(str(out / "prefix.mid"))

    print(f"\nAblation done. Results saved to {out}/")
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: compare SSM types (none/random/affinity/empirical)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Путь к чекпоинту SingLS (.pt)")
    parser.add_argument("--prompt", required=True,
                        help='Текстовый промт для MusicGen')
    parser.add_argument("--segment", required=True,
                        help='Сегментный план: "intro:8,verse:16,chorus:16,outro:8"')
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--n_gen_bars", type=int, default=64)
    parser.add_argument("--empirical_ckpt",
                        default="Seg2SSM/checkpoints/affinity_matrix.pt")
    parser.add_argument("--out_dir", default="outputs/ablation")

    args = parser.parse_args()
    segment_plan = parse_segment_plan(args.segment)

    run_ablation(
        model_path=args.model,
        prompt=args.prompt,
        segment_plan=segment_plan,
        tempo=args.tempo,
        n_gen_bars=args.n_gen_bars,
        empirical_ckpt=args.empirical_ckpt,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
