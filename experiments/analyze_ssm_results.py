"""
analyze_ssm_results.py — Статистический анализ SSM-following эксперимента.

Загружает сохранённый ssm_following_results.npz и вычисляет:
  - Парный t-тест (scipy.stats.ttest_rel)
  - Критерий Вилкоксона (непараметрическая альтернатива)
  - Cohen's d (размер эффекта)
  - 95% доверительный интервал для ΔIOU и ΔMSE
  - Bar chart с CI и значками значимости

Запуск:
  python -m experiments.analyze_ssm_results \
      --npz experiments/ssm_following_results/ssm_following_results.npz \
      --out_dir experiments/ssm_following_results
"""

import argparse
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ── Статистика ────────────────────────────────────────────────────────────────

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.mean() / diff.std())


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compute_stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    t, p_t  = stats.ttest_rel(a, b)
    w, p_w  = stats.wilcoxon(diff)
    d       = cohens_d_paired(a, b)
    ci      = stats.t.interval(0.95, df=len(diff) - 1,
                               loc=diff.mean(), scale=stats.sem(diff))
    n_better = int((diff > 0).sum())
    return {
        "mean_a":    float(a.mean()),
        "std_a":     float(a.std()),
        "mean_b":    float(b.mean()),
        "std_b":     float(b.std()),
        "delta":     float(diff.mean()),
        "delta_std": float(diff.std()),
        "t":         float(t),
        "p_t":       float(p_t),
        "W":         float(w),
        "p_w":       float(p_w),
        "d":         float(d),
        "ci_lo":     float(ci[0]),
        "ci_hi":     float(ci[1]),
        "n_better":  n_better,
        "n_total":   len(a),
    }


# ── Отчёт ─────────────────────────────────────────────────────────────────────

def format_p(p: float) -> str:
    if p < 0.0001:
        return "p<0.0001"
    return f"p={p:.4f}"


def build_report(iou: dict, mse: dict, n: int, npz_path: str) -> str:
    lines = [
        "SSM-Following Statistical Report",
        f"Generated: {datetime.date.today()}",
        f"Source:    {npz_path}",
        f"N={n} paired observations (same segment plan + prefix, two SSM conditions)",
        "",
        "=" * 60,
        "IOU ↑  (higher is better for affinity)",
        "=" * 60,
        f"  affinity : {iou['mean_a']:.4f} ± {iou['std_a']:.4f}",
        f"  none     : {iou['mean_b']:.4f} ± {iou['std_b']:.4f}",
        f"  ΔIOU     : {iou['delta']:+.4f}",
        f"  Paired t-test : t={iou['t']:.3f}, {format_p(iou['p_t'])} {significance_stars(iou['p_t'])}",
        f"  Wilcoxon      : W={iou['W']:.0f},  {format_p(iou['p_w'])} {significance_stars(iou['p_w'])}",
        f"  Cohen's d     : {iou['d']:.3f}",
        f"  95% CI ΔIOU   : [{iou['ci_lo']:+.4f}, {iou['ci_hi']:+.4f}]",
        f"  Pairs affinity>none: {iou['n_better']}/{iou['n_total']}",
        "",
        "=" * 60,
        "MSE ↓  (lower is better for affinity)",
        "=" * 60,
        f"  affinity : {mse['mean_a']:.4f} ± {mse['std_a']:.4f}",
        f"  none     : {mse['mean_b']:.4f} ± {mse['std_b']:.4f}",
        f"  ΔMSE     : {mse['delta']:+.4f}",
        f"  Paired t-test : t={mse['t']:.3f}, {format_p(mse['p_t'])} {significance_stars(mse['p_t'])}",
        f"  Wilcoxon      : W={mse['W']:.0f},  {format_p(mse['p_w'])} {significance_stars(mse['p_w'])}",
        f"  Cohen's d     : {mse['d']:.3f}",
        f"  95% CI ΔMSE   : [{mse['ci_lo']:+.4f}, {mse['ci_hi']:+.4f}]",
        f"  Pairs affinity<none: {mse['n_better']}/{mse['n_total']}",
        "",
        "=" * 60,
        "LaTeX table snippet",
        "=" * 60,
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Condition & IOU $\uparrow$ & MSE $\downarrow$ \\",
        r"\hline",
        f"affinity & {iou['mean_a']:.4f}$\\pm${iou['std_a']:.4f} & {mse['mean_a']:.4f}$\\pm${mse['std_a']:.4f} \\\\",
        f"none     & {iou['mean_b']:.4f}$\\pm${iou['std_b']:.4f} & {mse['mean_b']:.4f}$\\pm${mse['std_b']:.4f} \\\\",
        r"\hline",
        f"$\\Delta$ & {iou['delta']:+.4f} & {mse['delta']:+.4f} \\\\",
        f"paired $t$ & {iou['t']:.2f} & {mse['t']:.2f} \\\\",
        f"$p$ & {format_p(iou['p_t'])} & {format_p(mse['p_t'])} \\\\",
        f"Cohen $d$ & {iou['d']:.3f} & {mse['d']:.3f} \\\\",
        r"\hline",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


# ── Визуализация ──────────────────────────────────────────────────────────────

def plot_bar_chart(
    iou_stats: dict,
    mse_stats: dict,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, s, metric, label, direction in [
        (axes[0], iou_stats, "IOU", "IOU ↑", "higher"),
        (axes[1], mse_stats, "MSE", "MSE ↓", "lower"),
    ]:
        means  = [s["mean_a"], s["mean_b"]]
        errors = [
            s["mean_a"] - s["ci_lo"] + s["delta"],   # CI halfwidth ≈ t*se
            s["std_b"] * 1.96 / np.sqrt(s["n_total"]),
        ]
        # Simpler: use std/sqrt(N) as SEM bars
        sems = [
            s["std_a"] / np.sqrt(s["n_total"]),
            s["std_b"] / np.sqrt(s["n_total"]),
        ]
        colors = ["steelblue", "salmon"]
        bars = ax.bar(["affinity", "none"], means, color=colors,
                      width=0.5, alpha=0.85)
        ax.errorbar(["affinity", "none"], means, yerr=[1.96 * se for se in sems],
                    fmt="none", color="black", capsize=5, linewidth=1.5)

        # Значок значимости
        stars = significance_stars(s["p_t"])
        y_max = max(means) + max(1.96 * se for se in sems)
        y_line = y_max * 1.06
        ax.plot([0, 1], [y_line, y_line], color="black", linewidth=1)
        ax.text(0.5, y_line * 1.01, stars, ha="center", va="bottom", fontsize=13)

        ax.set_title(f"{label}\n"
                     f"t={s['t']:.2f}, {format_p(s['p_t'])}, d={s['d']:.3f}",
                     fontsize=10)
        ax.set_ylabel(metric)
        ax.set_ylim(0, y_line * 1.12)

    plt.suptitle(
        f"SSM-Following: affinity vs. none  (N={iou_stats['n_total']}, paired t-test)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Bar chart  : {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Статистический анализ SSM-following эксперимента",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--npz", required=True,
        help="Путь к ssm_following_results.npz",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Директория для вывода (по умолчанию: директория --npz)",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_dir  = Path(args.out_dir) if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    aff_iou  = data["affinity_iou"]
    none_iou = data["none_iou"]
    aff_mse  = data["affinity_mse"]
    none_mse = data["none_mse"]

    iou_stats = compute_stats(aff_iou, none_iou)
    # Для MSE "лучше" значит affinity < none → diff = none - affinity (чтоб n_better считать верно)
    mse_stats_raw = compute_stats(aff_mse, none_mse)
    # Перекодируем n_better для MSE: affinity<none ↔ diff<0
    mse_stats = dict(mse_stats_raw)
    mse_stats["n_better"] = int((aff_mse < none_mse).sum())

    report = build_report(iou_stats, mse_stats, len(aff_iou), str(npz_path))
    print(report)

    report_path = out_dir / "ssm_stats_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nОтчёт сохранён: {report_path}")

    chart_path = out_dir / "ssm_stats_barchart.png"
    plot_bar_chart(iou_stats, mse_stats_raw, chart_path)


if __name__ == "__main__":
    main()
