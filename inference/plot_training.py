"""
Утилита для построения графиков обучения из CSV-логов.

Запуск:
    # Один прогон
    python3 -m inference.plot_training --csv data/logs/my_model_metrics.csv

    # Несколько прогонов на одном графике (для сравнения)
    python3 -m inference.plot_training \\
        --csv data/logs/run_a_metrics.csv data/logs/run_b_metrics.csv \\
        --labels "SING λ=1" "SING λ=5" \\
        --out outputs/comparison.png
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


_LOSS_COMPONENTS = ["total_loss", "bce_loss", "ssm_loss", "struct_loss"]
_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


def plot_training_curves(
    csv_paths: List[str],
    labels: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Строит 4 subplot-а: total / bce / ssm / struct loss по эпохам.

    Args:
        csv_paths : список путей к CSV-файлам с колонками _CSV_COLUMNS
        labels    : подписи кривых (по умолчанию — имена файлов)
        out_path  : куда сохранить PNG; если None — сохраняет рядом с первым CSV
        show      : показывать интерактивно (не работает без дисплея)
    """
    if labels is None:
        labels = [Path(p).stem.replace("_metrics", "") for p in csv_paths]

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        dfs.append(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    titles = ["Total Loss", "BCE Loss", "SSM Loss", "Struct Loss"]
    cols   = _LOSS_COMPONENTS

    for ax, col, title in zip(axes, cols, titles):
        for df, label, color in zip(dfs, labels, _COLORS):
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col], label=label, color=color, linewidth=1.8)
            ax.scatter(df["epoch"], df[col], s=15, color=color, zorder=3)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Training Loss Curves", fontsize=14)
    plt.tight_layout()

    if out_path is None:
        out_path = str(Path(csv_paths[0]).with_suffix(".png"))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved → {out_path}")

    if show:
        plt.show()


def print_summary(csv_path: str) -> None:
    """Печатает текстовую сводку по CSV-логу."""
    df = pd.read_csv(csv_path)
    print(f"\n{'=' * 55}")
    print(f"Run: {Path(csv_path).stem}")
    print(f"  Epochs logged  : {len(df)}")
    for col in _LOSS_COMPONENTS:
        if col not in df.columns:
            continue
        best_epoch = int(df.loc[df[col].idxmin(), "epoch"])
        best_val   = df[col].min()
        last_val   = df[col].iloc[-1]
        print(f"  {col:<14}: final={last_val:.5f}  best={best_val:.5f} (epoch {best_epoch})")
    total_time = df["train_time_s"].sum() if "train_time_s" in df.columns else None
    if total_time:
        print(f"  Total time     : {total_time / 60:.1f} min")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves from CSV logs")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV-файл(ы) с метриками")
    parser.add_argument("--labels", nargs="+", default=None, help="Подписи кривых")
    parser.add_argument("--out", default=None, help="Путь для сохранения PNG")
    parser.add_argument("--summary", action="store_true", help="Печатать текстовую сводку")
    args = parser.parse_args()

    if args.summary or len(args.csv) == 1:
        for p in args.csv:
            print_summary(p)

    plot_training_curves(args.csv, labels=args.labels, out_path=args.out)
