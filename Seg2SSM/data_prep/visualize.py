"""
Быстрая проверка: визуализируем N случайных пар (block_ssm, gt_ssm) из датасета.

Запуск:
    python visualize.py --dataset ../../data/seg2ssm/seg2ssm_train.pt --n 6
"""

import argparse
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


LABEL_NAMES = {0: "intro", 1: "verse", 2: "chorus", 3: "bridge",
               4: "instr", 5: "outro", 6: "other"}
LABEL_COLORS = {0: "#888", 1: "#4C9BE8", 2: "#E84C4C", 3: "#F0A500",
                4: "#7DB87D", 5: "#9B7DB8", 6: "#ccc"}


def draw_segment_bar(ax, segment_plan, width=64):
    """Рисует горизонтальную полоску с цветными секциями."""
    total = sum(d for _, d in segment_plan)
    x = 0
    for label_id, dur in segment_plan:
        frac = dur / total
        color = LABEL_COLORS.get(label_id, "#ccc")
        ax.barh(0, frac * width, left=x * width / total, height=1, color=color, edgecolor="white")
        x += dur
    ax.set_xlim(0, width)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--output", default="preview.png")
    args = parser.parse_args()

    data = torch.load(args.dataset, weights_only=False)
    examples = data["examples"]
    samples = random.sample(examples, min(args.n, len(examples)))

    fig, axes = plt.subplots(args.n, 3, figsize=(12, args.n * 2.5))
    if args.n == 1:
        axes = [axes]

    for row, ex in enumerate(samples):
        block = ex["block_ssm"].numpy()
        gt = ex["gt_ssm"].numpy()
        seg = ex["segment_plan"]

        # Блочная SSM
        axes[row][0].imshow(block, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axes[row][0].set_title(f"{ex['song_id']} | block SSM", fontsize=8)
        axes[row][0].axis("off")

        # GT SSM
        axes[row][1].imshow(gt, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axes[row][1].set_title(f"GT SSM (shift={ex.get('pitch_shift',0)})", fontsize=8)
        axes[row][1].axis("off")

        # Сегментный план
        draw_segment_bar(axes[row][2], seg)
        labels = [LABEL_NAMES.get(l, "?") for l, _ in seg]
        axes[row][2].set_title("  ".join(f"{l}({d})" for l, d in seg[:6]), fontsize=7)

    # Легенда
    patches = [mpatches.Patch(color=c, label=n) for n, c in
               zip(LABEL_NAMES.values(), LABEL_COLORS.values())]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
