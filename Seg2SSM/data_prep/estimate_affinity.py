"""
Шаг 1: Эмпирическая оценка Affinity Matrix из SALAMI данных.

Алгоритм:
  Для каждого трека:
    1. Загружаем bar_times + chroma из features/
    2. Выравниваем сегменты → bar_labels [T]
    3. Вычисляем tanh-нормализованную SSM [T, T]
    4. Для каждой пары баров (i, j) накапливаем SSM[i,j]
       по ключу (label_i, label_j)
  Итог: A[a, b] = mean(SSM[i,j]) для всех пар с метками (a, b)

Запуск:
    python estimate_affinity.py \
        --annotations annotations.json \
        --features_dir features/ \
        --output_dir ../checkpoints/
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Импортируем уже написанные функции
from build_dataset import (
    align_segments_to_bars,
    chroma_to_ssm,
)

LABEL_NAMES = {
    0: "intro",
    1: "verse",
    2: "chorus",
    3: "bridge",
    4: "instr",
    5: "outro",
    6: "other",
    7: "unknown",
}
N_LABELS = len(LABEL_NAMES)


def estimate_affinity(annotations_path: str, features_dir: str) -> np.ndarray:
    """
    Возвращает матрицу A [N_LABELS, N_LABELS], где
    A[a, b] = среднее значение GT_SSM по всем парам баров (i,j)
    с метками label_i=a, label_j=b.
    """
    with open(annotations_path) as f:
        data = json.load(f)

    features_dir = Path(features_dir)

    # Накапливаем сумму и количество по парам меток
    sums   = np.zeros((N_LABELS, N_LABELS), dtype=np.float64)
    counts = np.zeros((N_LABELS, N_LABELS), dtype=np.int64)

    processed = 0
    for track in data["tracks"]:
        feat_path = features_dir / f"{track['song_id']}.pt"
        if not feat_path.exists():
            continue

        features = torch.load(feat_path, weights_only=True)
        chroma    = features["chroma"]    # [T, 12]
        bar_times = features["bar_times"]
        T = chroma.shape[0]

        if T < 8:
            continue

        aligned = align_segments_to_bars(track["segments"], bar_times)
        if not aligned:
            continue

        # Обрезаем до T
        for seg in aligned:
            seg["end_bar"] = min(seg["end_bar"], T)
        aligned = [s for s in aligned if s["duration_bars"] > 0]
        if not aligned:
            continue

        # Строим bar_labels
        bar_labels = np.full(T, fill_value=7, dtype=np.int64)  # 7 = unknown
        for seg in aligned:
            s, e = seg["start_bar"], seg["end_bar"]
            bar_labels[s:e] = seg["label_id"]

        # SSM с tanh-нормализацией
        ssm = chroma_to_ssm(chroma).numpy()  # [T, T]

        # Накапливаем по парам меток (только верхний треугольник + диагональ)
        for i in range(T):
            la = bar_labels[i]
            for j in range(i, T):
                lb = bar_labels[j]
                v  = ssm[i, j]
                sums[la, lb]   += v
                counts[la, lb] += 1
                if i != j:
                    sums[lb, la]   += v
                    counts[lb, la] += 1

        processed += 1

    print(f"Processed {processed} tracks")

    # Среднее; где нет данных — ставим 0.5 (нейтральное)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(counts > 0, sums / counts, 0.5)

    return A.astype(np.float32), counts


def plot_affinity(A: np.ndarray, counts: np.ndarray, output_path: str):
    """Визуализирует матрицу A как heatmap с аннотациями."""
    labels = [LABEL_NAMES[i] for i in range(N_LABELS)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Affinity heatmap ---
    ax = axes[0]
    im = ax.imshow(A, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(N_LABELS))
    ax.set_yticks(range(N_LABELS))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Affinity Matrix A\n(среднее SSM по парам секций)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(N_LABELS):
        for j in range(N_LABELS):
            ax.text(j, i, f"{A[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if A[i,j] > 0.65 else "black",
                    fontsize=8)

    # --- Count heatmap (сколько пар использовалось) ---
    ax2 = axes[1]
    log_counts = np.log10(counts + 1)
    im2 = ax2.imshow(log_counts, cmap="Greens")
    ax2.set_xticks(range(N_LABELS))
    ax2.set_yticks(range(N_LABELS))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_title("Количество пар (log10)\n(надёжность оценки)", fontsize=12)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    for i in range(N_LABELS):
        for j in range(N_LABELS):
            ax2.text(j, i, f"{int(counts[i,j]):,}".replace(",", " "),
                     ha="center", va="center",
                     color="white" if log_counts[i,j] > 3 else "black",
                     fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Heatmap → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations",  default="annotations.json")
    parser.add_argument("--features_dir", default="features/")
    parser.add_argument("--output_dir",   default="../checkpoints/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Estimating affinity matrix from data...")
    A, counts = estimate_affinity(args.annotations, args.features_dir)

    print("\nAffinity Matrix A:")
    labels = [LABEL_NAMES[i] for i in range(N_LABELS)]
    header = f"{'':>8}" + "".join(f"{l:>8}" for l in labels)
    print(header)
    for i, la in enumerate(labels):
        row = f"{la:>8}" + "".join(f"{A[i,j]:>8.3f}" for j in range(N_LABELS))
        print(row)

    # Сохраняем
    ckpt_path = output_dir / "affinity_matrix.pt"
    torch.save({
        "A":          torch.from_numpy(A),
        "counts":     torch.from_numpy(counts),
        "label_names": LABEL_NAMES,
    }, ckpt_path)
    print(f"\nSaved → {ckpt_path}")

    # Визуализация
    plot_affinity(A, counts, str(output_dir / "affinity_heatmap.png"))


if __name__ == "__main__":
    main()
