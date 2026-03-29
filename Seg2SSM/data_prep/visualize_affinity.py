"""
Визуализация Affinity-based SSM для презентации.

Генерирует три картинки:
  1. affinity_comparison.png  — сравнение A_fixed vs A_empirical
  2. ssm_examples.png         — примеры: block_ssm | affinity_ssm | gt_ssm
  3. segment_gallery.png      — affinity SSM для разных типичных форм песен

Запуск:
    python visualize_affinity.py \
        --dataset    ../../data/seg2ssm/seg2ssm_train.pt \
        --checkpoint ../checkpoints/affinity_matrix.pt \
        --output_dir ../eval_results/affinity/
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
from affinity_ssm import AffinitySSM, LABEL_NAMES, N_LABELS

LABEL_COLORS = {
    0: "#888888",   # intro
    1: "#4C9BE8",   # verse
    2: "#E84C4C",   # chorus
    3: "#F0A500",   # bridge
    4: "#7DB87D",   # instr
    5: "#9B7DB8",   # outro
    6: "#cccccc",   # other
    7: "#eeeeee",   # unknown
}


# ---------------------------------------------------------------------------
# 1. Сравнение A_fixed vs A_empirical
# ---------------------------------------------------------------------------

def plot_matrix_comparison(A_fixed: torch.Tensor, A_emp: torch.Tensor,
                            output_path: str):
    labels = [LABEL_NAMES[i] for i in range(N_LABELS)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, A, title in zip(
        axes,
        [A_fixed.numpy(), A_emp.numpy()],
        ["A_fixed  (теория музыки)", "A_empirical  (из 422 треков SALAMI)"],
    ):
        im = ax.imshow(A, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(N_LABELS))
        ax.set_yticks(range(N_LABELS))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=12, pad=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(N_LABELS):
            for j in range(N_LABELS):
                ax.text(j, i, f"{A[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if A[i, j] > 0.62 else "black",
                        fontsize=8)

    plt.suptitle("Матрица музыкального сходства A", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Comparison → {output_path}")


# ---------------------------------------------------------------------------
# 2. block_ssm | affinity_ssm | gt_ssm  для N примеров из датасета
# ---------------------------------------------------------------------------

def draw_segment_strip(ax, segment_plan, width=64):
    """Горизонтальная полоска с цветными секциями."""
    total = sum(d for _, d in segment_plan)
    x = 0
    for label_id, dur in segment_plan:
        frac = dur / total
        color = LABEL_COLORS.get(label_id, "#ccc")
        ax.barh(0, frac * width, left=x * width / total,
                height=1, color=color, edgecolor="white", linewidth=0.5)
        x += dur
    ax.set_xlim(0, width)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")


def plot_ssm_examples(dataset_path: str, builder_fixed: AffinitySSM,
                      builder_emp: AffinitySSM, output_path: str,
                      n: int = 6, seed: int = 42):
    random.seed(seed)
    data = torch.load(dataset_path, weights_only=False)
    examples = data["examples"]
    samples = random.sample(examples, min(n, len(examples)))

    # Колонки: segment strip | block_ssm | affinity_fixed | affinity_emp | gt_ssm
    col_titles = ["Сегментный план", "Block SSM\n(вход)", "Affinity SSM\nfixed",
                  "Affinity SSM\nempirical", "GT SSM\n(аудио chroma)"]
    ncols = len(col_titles)
    fig = plt.figure(figsize=(ncols * 3.2, n * 3.0))
    outer = gridspec.GridSpec(n, ncols, figure=fig, hspace=0.35, wspace=0.15)

    for row, ex in enumerate(samples):
        seg   = ex["segment_plan"]
        block = ex["block_ssm"]    # [64,64] или [1,64,64]
        gt    = ex["gt_ssm"]

        if block.dim() == 3:
            block = block.squeeze(0)
        if gt.dim() == 3:
            gt = gt.squeeze(0)

        # Affinity SSMs
        T_total = sum(d for _, d in seg)
        af_fixed = builder_fixed.build(seg, ssm_size=64)
        af_emp   = builder_emp.build(seg, ssm_size=64)

        images = [None, block, af_fixed, af_emp, gt]

        for col in range(ncols):
            ax = fig.add_subplot(outer[row, col])
            if col == 0:
                draw_segment_strip(ax, seg)
                plan_str = "  ".join(
                    f"{LABEL_NAMES.get(l,'?')}({d})" for l, d in seg[:5]
                )
                ax.set_title(f"{ex['song_id']}\n{plan_str}", fontsize=6.5)
            else:
                img = images[col].numpy() if hasattr(images[col], "numpy") \
                      else np.array(images[col])
                ax.imshow(img, cmap="Blues", vmin=0, vmax=1, aspect="auto")
                ax.axis("off")
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=9)

    # Легенда секций
    patches = [mpatches.Patch(color=c, label=LABEL_NAMES[i])
               for i, c in LABEL_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=N_LABELS,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Examples → {output_path}")


# ---------------------------------------------------------------------------
# 3. Галерея типичных форм песен
# ---------------------------------------------------------------------------

SONG_FORMS = {
    "Verse-Chorus": [
        (0, 8), (1, 16), (2, 16), (1, 16), (2, 16), (5, 8),
    ],
    "Verse-Chorus-Bridge": [
        (0, 8), (1, 16), (2, 16), (1, 16), (2, 16), (3, 8), (2, 16), (5, 8),
    ],
    "AABA (jazz)": [
        (1, 16), (1, 16), (3, 16), (1, 16),
    ],
    "Intro-Verse-Pre-Chorus": [
        (0, 8), (1, 16), (4, 8), (2, 16), (1, 16), (4, 8), (2, 16), (5, 8),
    ],
    "Стандартная поп-форма": [
        (0, 4), (1, 16), (2, 16), (1, 16), (2, 16),
        (3, 8), (2, 16), (2, 16), (5, 8),
    ],
}


def plot_segment_gallery(builder_fixed: AffinitySSM, builder_emp: AffinitySSM,
                         output_path: str):
    forms = list(SONG_FORMS.items())
    n = len(forms)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.2))

    col_titles = ["Сегментный план", "Affinity SSM (fixed)", "Affinity SSM (empirical)"]

    for row, (name, plan) in enumerate(forms):
        af_fixed = builder_fixed.build(plan, ssm_size=64).numpy()
        af_emp   = builder_emp.build(plan, ssm_size=64).numpy()

        # Сегментный план
        ax0 = axes[row][0]
        draw_segment_strip(ax0, plan)
        plan_str = " → ".join(LABEL_NAMES.get(l, "?") for l, _ in plan)
        ax0.set_title(f"{name}\n{plan_str}", fontsize=8)

        # Affinity SSM fixed
        axes[row][1].imshow(af_fixed, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axes[row][1].axis("off")

        # Affinity SSM empirical
        axes[row][2].imshow(af_emp, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axes[row][2].axis("off")

        if row == 0:
            for ax, t in zip(axes[0], col_titles):
                ax.set_title(t, fontsize=10)

    patches = [mpatches.Patch(color=c, label=LABEL_NAMES[i])
               for i, c in LABEL_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=N_LABELS,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Affinity SSM для типичных форм песен", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Gallery → {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="../../data/seg2ssm/seg2ssm_train.pt")
    parser.add_argument("--checkpoint", default="../checkpoints/affinity_matrix.pt")
    parser.add_argument("--output_dir", default="../eval_results/affinity/")
    parser.add_argument("--n",          type=int, default=6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder_fixed = AffinitySSM.fixed()
    builder_emp   = AffinitySSM.from_checkpoint(args.checkpoint)

    # 1. Сравнение матриц
    plot_matrix_comparison(
        builder_fixed.A, builder_emp.A,
        str(output_dir / "affinity_comparison.png"),
    )

    # 2. Примеры из датасета
    plot_ssm_examples(
        args.dataset, builder_fixed, builder_emp,
        str(output_dir / "ssm_examples.png"),
        n=args.n,
    )

    # 3. Галерея типичных форм
    plot_segment_gallery(
        builder_fixed, builder_emp,
        str(output_dir / "segment_gallery.png"),
    )

    print(f"\nВсе визуализации → {output_dir}")


if __name__ == "__main__":
    main()
