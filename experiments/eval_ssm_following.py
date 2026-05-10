"""
eval_ssm_following.py — Оценка SSM-following rate.

Насколько хорошо модель следует AffinitySSM vs. нулевой SSM (baseline)?

Алгоритм:
  1. Генерируем N случайных сегментных планов (SALAMI-стиль).
  2. Для каждого плана строим AffinitySSM как целевую структуру.
  3. Генерируем трек дважды: с affinity-SSM и без (none).
  4. Вычисляем chroma-SSM из сгенерированного piano roll.
  5. Считаем IOU и MSE между target SSM и actual SSM.
  6. Выводим итоговую таблицу и гистограммы.

Запуск:
  python -m experiments.eval_ssm_following \
      --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
      --dataset data/combined/combined_test.pt \
      --n_plans 100 \
      --n_gen_bars 64 \
      --out_dir experiments/ssm_following_results
"""

import argparse
import datetime
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

from SingLS.config.config import DEVICE
from SingLS.trainer.data_utils import SSM as chroma_ssm
from Seg2SSM.affinity_ssm import AffinitySSM, LABEL_NAMES
from pipeline.generate import generate_from_prefix, build_ssm, SSMType, BEATS_PER_BAR
from inference.compare_models import iou_ssm, mse_ssm


LABEL_NAME_TO_ID = {v: k for k, v in LABEL_NAMES.items()}

# Параметры генерации совпадают с дефолтом pipeline/generate.py,
# чтобы eval измерял то же поведение, что видно в реальных запусках.
# n_samples=3: 3 one-hot OR → ~3 ноты/бит, density≈0.023
# (n_samples=1 из SAMPLING_EXPERIMENTS — для аудио-качества, не для eval)
GEN_PARAMS = dict(
    warmup_beats=16,
    temperature=1.5,
    temp_start=3.0,
    temp_warmup_steps=30,
    topk_k=50,
    n_samples=3,
    prefix_in_sequence=False,
    use_softmax=True,
)

PREFIX_BEATS = 16   # 4 бара из датасета — достаточно для прогрева LSTM


# ── Загрузка prefix-фрагментов из датасета ───────────────────────────────────

def load_prefixes(dataset_path: str, n_beats: int, rng: random.Random) -> List[torch.Tensor]:
    """
    Загружает датасет и возвращает список prefix-тензоров [n_beats, 128].
    Каждый — случайный фрагмент из случайного трека датасета.
    Треки короче n_beats пропускаются.
    """
    print(f"Загрузка датасета для prefix: {dataset_path}")
    data = torch.load(dataset_path, weights_only=False, map_location="cpu")
    prefixes = []
    indices = list(range(len(data)))
    rng.shuffle(indices)
    for idx in indices:
        item = data[idx]
        # Формат combined: numpy.ndarray shape (3,) = [piano_roll, tempo, num_beats]
        # Формат MAESTRO/LMD: list/tuple (piano_roll, tempo, num_beats)
        if isinstance(item, (list, tuple)) or (hasattr(item, '__len__') and len(item) == 3 and not torch.is_tensor(item)):
            roll = item[0]
        else:
            roll = item  # raw tensor
        if not torch.is_tensor(roll):
            roll = torch.tensor(roll)
        roll = roll.float()
        if roll.shape[0] < n_beats:
            continue
        start = rng.randint(0, roll.shape[0] - n_beats)
        prefixes.append(roll[start: start + n_beats].float())
    print(f"  Доступно prefix-фрагментов: {len(prefixes)}\n")
    return prefixes


# ── Генерация случайных сегментных планов ────────────────────────────────────

def random_segment_plan(rng: random.Random) -> List[Tuple[int, int]]:
    """
    Генерирует реалистичный сегментный план в стиле SALAMI.
    Структура: intro → [verse → chorus]×K → [bridge →] [verse → chorus]×L → outro
    """
    plan: List[Tuple[int, int]] = []

    plan.append((LABEL_NAME_TO_ID["intro"], rng.choice([4, 8])))

    K = rng.randint(1, 3)
    for _ in range(K):
        plan.append((LABEL_NAME_TO_ID["verse"],  rng.choice([12, 16])))
        plan.append((LABEL_NAME_TO_ID["chorus"], rng.choice([8, 12, 16])))

    if rng.random() < 0.5:
        plan.append((LABEL_NAME_TO_ID["bridge"], rng.choice([8, 12])))

    L = rng.randint(1, 2)
    for _ in range(L):
        plan.append((LABEL_NAME_TO_ID["verse"],  rng.choice([12, 16])))
        plan.append((LABEL_NAME_TO_ID["chorus"], rng.choice([8, 12, 16])))

    plan.append((LABEL_NAME_TO_ID["outro"], rng.choice([4, 8])))
    return plan


def plan_to_str(plan: List[Tuple[int, int]]) -> str:
    return ",".join(f"{LABEL_NAMES[lbl]}:{bars}" for lbl, bars in plan)


# ── Генерация трека ──────────────────────────────────────────────────────────

def generate_track(
    model,
    ssm_type: SSMType,
    segment_plan: List[Tuple[int, int]],
    n_gen_bars: int,
    prefix: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Генерирует n_gen_bars тактов с заданным SSM.

    prefix — фрагмент [PREFIX_BEATS, 128] из датасета (in-distribution).
    Если не передан, используется нулевой тензор (холодный старт).

    Returns:
        piano_roll [T_gen, 128]  (только сгенерированная часть, без prefix)
    """
    T_prefix = PREFIX_BEATS
    T_gen    = n_gen_bars * BEATS_PER_BAR
    T_total  = T_prefix + T_gen

    if prefix is None:
        prefix = torch.zeros(T_prefix, 128)

    ssm = build_ssm(ssm_type, segment_plan, T_total).to(DEVICE)

    sequence  = generate_from_prefix(model, prefix, ssm, T_gen, **GEN_PARAMS)
    full_roll = sequence.squeeze(1).detach().cpu().numpy().round()  # [T_total, 128]

    # При prefix_in_sequence=False: first T_prefix frames = prefix, затем T_gen gen.
    return full_roll[T_prefix: T_prefix + T_gen]


# ── SSM-метрики ──────────────────────────────────────────────────────────────

def piano_roll_to_ssm(piano_roll: np.ndarray, target_size: int) -> np.ndarray:
    """
    Chroma-based SSM из piano roll [T, 128] → [target_size, target_size].
    Если T != target_size, выполняет bilinear resize.
    """
    roll_t = torch.tensor(piano_roll, dtype=torch.float32)
    ssm = chroma_ssm(roll_t)          # [T, T]
    T = ssm.shape[0]
    if T != target_size:
        ssm = F.interpolate(
            ssm.unsqueeze(0).unsqueeze(0).float(),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return ssm.cpu().numpy()


# ── Основной цикл оценки ─────────────────────────────────────────────────────

def evaluate_plan(
    model,
    segment_plan: List[Tuple[int, int]],
    n_gen_bars: int,
    ssm_size: int,
    prefix: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Оценивает один план для двух условий (affinity / none).
    Оба условия используют одинаковый prefix для честного сравнения.
    Возвращает: {"affinity": {iou, mse, note_density}, "none": {...}}

    target_ssm — чистая блочная AffinitySSM БЕЗ шума и rescaling:
      это «структурное намерение» пользователя, против которого измеряем.
    Модель получает реалистичную версию (с шумом и rescaling) — через build_ssm().
    """
    # Нет шума, нет rescaling → чистый блочный паттерн = структурное намерение
    target_ssm = AffinitySSM.fixed(noise_std=0.0, rescale=False).build(
        segment_plan, ssm_size=ssm_size
    ).numpy()

    results: Dict[str, Dict[str, float]] = {}
    for ssm_type in (SSMType.AFFINITY, SSMType.NONE):
        roll       = generate_track(model, ssm_type, segment_plan, n_gen_bars, prefix=prefix)
        actual_ssm = piano_roll_to_ssm(roll, ssm_size)
        results[ssm_type.value] = {
            "iou":          float(iou_ssm(target_ssm, actual_ssm)),
            "mse":          float(mse_ssm(target_ssm, actual_ssm)),
            "note_density": float(roll.mean()),
        }
    return results


# ── Визуализация ─────────────────────────────────────────────────────────────

def plot_histograms(
    all_metrics: Dict[str, Dict[str, List[float]]],
    n_plans: int,
    n_gen_bars: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = {"affinity": "steelblue", "none": "salmon"}

    for ax, (metric, label, better) in zip(
        axes,
        [("iou", "IOU ↑", "higher is better"),
         ("mse", "MSE ↓", "lower is better")],
    ):
        for cond, color in palette.items():
            vals = all_metrics[cond][metric]
            mu   = np.mean(vals)
            ax.hist(vals, bins=20, alpha=0.65, color=color,
                    label=f"{cond} (μ={mu:.4f})")
        ax.set_xlabel(label)
        ax.set_ylabel("Число планов")
        ax.set_title(f"{label} ({better})")
        ax.legend()

    plt.suptitle(
        f"SSM-Following Rate | N={n_plans} планов, {n_gen_bars} тактов",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_example_ssms(
    model,
    plan: List[Tuple[int, int]],
    n_gen_bars: int,
    ssm_size: int,
    out_path: Path,
    prefix: Optional[torch.Tensor] = None,
) -> None:
    """Сохраняет grid из 3 SSM для одного примера: target / affinity / none."""
    # target — чистый блочный паттерн (структурное намерение)
    target_ssm = AffinitySSM.fixed(noise_std=0.0, rescale=False).build(
        plan, ssm_size=ssm_size
    ).numpy()

    ssms: Dict[str, np.ndarray] = {"target\n(AffinitySSM)": target_ssm}
    for ssm_type in (SSMType.AFFINITY, SSMType.NONE):
        roll = generate_track(model, ssm_type, plan, n_gen_bars, prefix=prefix)
        ssms[f"actual\n({ssm_type.value})"] = piano_roll_to_ssm(roll, ssm_size)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, ssm) in zip(axes, ssms.items()):
        im = ax.imshow(ssm, origin="lower", cmap="hot", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plan_str = " → ".join(
        f"{LABEL_NAMES[l]}:{b}b" for l, b in plan
    )
    plt.suptitle(f"Пример SSM | {plan_str}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка SSM-following rate: affinity vs. none",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Путь к чекпоинту модели (.pt / .txt-zip), например:\n"
             "  data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt",
    )
    parser.add_argument(
        "--dataset", default="data/combined/combined_test.pt",
        help="Датасет для in-distribution prefix (по умолчанию: data/combined/combined_test.pt).\n"
             "Если файл не найден, используется нулевой prefix (холодный старт).",
    )
    parser.add_argument(
        "--n_plans", type=int, default=100,
        help="Число случайных сегментных планов (по умолчанию: 100)",
    )
    parser.add_argument(
        "--n_gen_bars", type=int, default=64,
        help="Такты генерации на план (по умолчанию: 64)",
    )
    parser.add_argument(
        "--ssm_size", type=int, default=64,
        help="Размер SSM для сравнения, NxN (по умолчанию: 64)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed для воспроизводимости (по умолчанию: 42)",
    )
    parser.add_argument(
        "--out_dir", default="experiments/ssm_following_results",
        help="Директория для результатов (по умолчанию: experiments/ssm_following_results/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Загрузка модели ───────────────────────────────────────────────────
    print(f"Загрузка модели: {args.model}")
    model = torch.load(args.model, weights_only=False, map_location=DEVICE)
    model.eval()
    print(f"Тип модели: {type(model).__name__}\n")

    # ── Загрузка prefix-фрагментов из датасета ───────────────────────────
    rng = random.Random(args.seed)
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        all_prefixes = load_prefixes(str(dataset_path), PREFIX_BEATS, rng)
    else:
        print(f"[!] Датасет не найден: {dataset_path} — используется нулевой prefix\n")
        all_prefixes = []

    def sample_prefix(i: int) -> Optional[torch.Tensor]:
        if not all_prefixes:
            return None
        return all_prefixes[i % len(all_prefixes)]

    # ── Генерация планов ──────────────────────────────────────────────────
    plans = [random_segment_plan(rng) for _ in range(args.n_plans)]

    # ── Сохраняем пример SSM до основного цикла ───────────────────────────
    print("Визуализация примера SSM (первый план)...")
    plot_example_ssms(
        model, plans[0], args.n_gen_bars, args.ssm_size,
        out_dir / "example_ssm_comparison.png",
        prefix=sample_prefix(0),
    )
    print(f"  Сохранено: {out_dir / 'example_ssm_comparison.png'}\n")

    # ── Основной цикл ────────────────────────────────────────────────────
    all_metrics: Dict[str, Dict[str, List[float]]] = {
        "affinity": {"iou": [], "mse": [], "note_density": []},
        "none":     {"iou": [], "mse": [], "note_density": []},
    }

    prefix_source = str(dataset_path) if all_prefixes else "zeros (cold start)"
    print(
        f"Оценка {args.n_plans} планов | "
        f"n_gen_bars={args.n_gen_bars} | ssm_size={args.ssm_size}×{args.ssm_size} | "
        f"prefix={prefix_source}"
    )
    print("-" * 60)

    for i, plan in enumerate(plans):
        total_bars = sum(b for _, b in plan)
        print(f"  [{i+1:3d}/{args.n_plans}] {plan_to_str(plan)}  ({total_bars} bars total)")

        results = evaluate_plan(
            model, plan, args.n_gen_bars, args.ssm_size,
            prefix=sample_prefix(i),
        )

        for cond in ("affinity", "none"):
            for metric in ("iou", "mse", "note_density"):
                all_metrics[cond][metric].append(results[cond][metric])

        # Промежуточный вывод каждые 10 планов
        if (i + 1) % 10 == 0:
            aff_iou  = np.mean(all_metrics["affinity"]["iou"])
            none_iou = np.mean(all_metrics["none"]["iou"])
            print(
                f"    [running avg] affinity IOU={aff_iou:.4f}  |  none IOU={none_iou:.4f}"
            )

    # ── Итоговая таблица ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SSM-Following Rate — Итог")
    print(f"N={args.n_plans} планов | n_gen_bars={args.n_gen_bars} | ssm_size={args.ssm_size}")
    print("=" * 65)
    print(f"{'Условие':>12}  {'IOU ↑':>14}  {'MSE ↓':>14}  {'NotesDensity':>12}")
    print("-" * 65)

    for cond in ("affinity", "none"):
        iou_arr = np.array(all_metrics[cond]["iou"])
        mse_arr = np.array(all_metrics[cond]["mse"])
        nd_arr  = np.array(all_metrics[cond]["note_density"])
        print(
            f"{cond:>12}  "
            f"{iou_arr.mean():.4f} ± {iou_arr.std():.4f}  "
            f"{mse_arr.mean():.4f} ± {mse_arr.std():.4f}  "
            f"{nd_arr.mean():.4f}"
        )

    print("=" * 65)

    # Дельта: насколько affinity лучше none
    delta_iou = np.mean(all_metrics["affinity"]["iou"]) - np.mean(all_metrics["none"]["iou"])
    delta_mse = np.mean(all_metrics["affinity"]["mse"]) - np.mean(all_metrics["none"]["mse"])
    print(f"\n  ΔIOU (affinity − none) = {delta_iou:+.4f}  {'✓ affinity лучше' if delta_iou > 0 else '✗ нет улучшения'}")
    print(f"  ΔMSE (affinity − none) = {delta_mse:+.4f}  {'✓ affinity лучше' if delta_mse < 0 else '✗ нет улучшения'}")

    # ── Статистические тесты ─────────────────────────────────────────────
    aff_iou_arr  = np.array(all_metrics["affinity"]["iou"])
    none_iou_arr = np.array(all_metrics["none"]["iou"])
    aff_mse_arr  = np.array(all_metrics["affinity"]["mse"])
    none_mse_arr = np.array(all_metrics["none"]["mse"])

    diff_iou = aff_iou_arr - none_iou_arr
    diff_mse = aff_mse_arr - none_mse_arr

    t_iou, p_iou = sp_stats.ttest_rel(aff_iou_arr, none_iou_arr)
    t_mse, p_mse = sp_stats.ttest_rel(aff_mse_arr, none_mse_arr)
    w_iou, pw_iou = sp_stats.wilcoxon(diff_iou)
    w_mse, pw_mse = sp_stats.wilcoxon(diff_mse)
    d_iou = float(diff_iou.mean() / diff_iou.std())
    d_mse = float(diff_mse.mean() / diff_mse.std())
    ci_iou = sp_stats.t.interval(0.95, df=len(diff_iou) - 1,
                                  loc=diff_iou.mean(), scale=sp_stats.sem(diff_iou))
    ci_mse = sp_stats.t.interval(0.95, df=len(diff_mse) - 1,
                                  loc=diff_mse.mean(), scale=sp_stats.sem(diff_mse))

    def _fmt_p(p: float) -> str:
        return "p<0.0001" if p < 0.0001 else f"p={p:.4f}"

    print("\n" + "=" * 65)
    print("Статистические тесты (парные)")
    print("=" * 65)
    print(f"  IOU: t={t_iou:.3f}, {_fmt_p(p_iou)}, Cohen's d={d_iou:.3f}")
    print(f"       Wilcoxon W={w_iou:.0f}, {_fmt_p(pw_iou)}")
    print(f"       95% CI ΔIOU: [{ci_iou[0]:+.4f}, {ci_iou[1]:+.4f}]")
    print(f"       Pairs affinity>none: {(diff_iou>0).sum()}/{len(diff_iou)}")
    print()
    print(f"  MSE: t={t_mse:.3f}, {_fmt_p(p_mse)}, Cohen's d={d_mse:.3f}")
    print(f"       Wilcoxon W={w_mse:.0f}, {_fmt_p(pw_mse)}")
    print(f"       95% CI ΔMSE: [{ci_mse[0]:+.4f}, {ci_mse[1]:+.4f}]")
    print(f"       Pairs affinity<none: {(diff_mse<0).sum()}/{len(diff_mse)}")

    # Сохраняем текстовый отчёт рядом с .npz
    stats_report_path = out_dir / "ssm_stats_report.txt"
    _report_lines = [
        "SSM-Following Statistical Report",
        f"Generated: {datetime.date.today()}",
        f"N={args.n_plans} paired observations",
        "",
        f"IOU: affinity={aff_iou_arr.mean():.4f}±{aff_iou_arr.std():.4f}, none={none_iou_arr.mean():.4f}±{none_iou_arr.std():.4f}",
        f"     ΔIOU={delta_iou:+.4f}",
        f"     Paired t-test: t={t_iou:.3f}, {_fmt_p(p_iou)}",
        f"     Wilcoxon:      W={w_iou:.0f}, {_fmt_p(pw_iou)}",
        f"     Cohen's d:     {d_iou:.3f}",
        f"     95% CI ΔIOU:   [{ci_iou[0]:+.4f}, {ci_iou[1]:+.4f}]",
        f"     Pairs affinity>none: {(diff_iou>0).sum()}/{len(diff_iou)}",
        "",
        f"MSE: affinity={aff_mse_arr.mean():.4f}±{aff_mse_arr.std():.4f}, none={none_mse_arr.mean():.4f}±{none_mse_arr.std():.4f}",
        f"     ΔMSE={delta_mse:+.4f}",
        f"     Paired t-test: t={t_mse:.3f}, {_fmt_p(p_mse)}",
        f"     Wilcoxon:      W={w_mse:.0f}, {_fmt_p(pw_mse)}",
        f"     Cohen's d:     {d_mse:.3f}",
        f"     95% CI ΔMSE:   [{ci_mse[0]:+.4f}, {ci_mse[1]:+.4f}]",
        f"     Pairs affinity<none: {(diff_mse<0).sum()}/{len(diff_mse)}",
    ]
    stats_report_path.write_text("\n".join(_report_lines), encoding="utf-8")
    print(f"\nСтат. отчёт : {stats_report_path}")

    # ── Сохранение результатов ────────────────────────────────────────────
    npz_path = out_dir / "ssm_following_results.npz"
    np.savez(npz_path, **{
        f"{cond}_{metric}": np.array(vals)
        for cond, m_dict in all_metrics.items()
        for metric, vals in m_dict.items()
    })
    print(f"\nДанные сохранены: {npz_path}")

    hist_path = out_dir / "ssm_following_histogram.png"
    plot_histograms(all_metrics, args.n_plans, args.n_gen_bars, hist_path)
    print(f"Гистограммы   : {hist_path}")
    print(f"Пример SSM    : {out_dir / 'example_ssm_comparison.png'}")


if __name__ == "__main__":
    main()
