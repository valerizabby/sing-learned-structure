"""
Быстрый замер IOU/MSE/ScaleCons для итерации между фиксами.
5 runs × 5 треков = 25 на модель (~2-4 мин).

Запуск:
    python3 -m inference.quick_eval
    python3 -m inference.quick_eval --label "Fix A: softmax"
"""

import argparse
import sys
import numpy as np
from collections import defaultdict
from inference.compare_models import compare_models_avg

DATA_PATH = "data/combined/combined_test.pt"

MODELS = {
    "SING (Original)": "data/meta_info/trained_original_combined/model_30_epochs.txt",
    "3SING*":          "data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt",
    "LSA":             "data/meta_info/trained_lsa_combined/model_30_epochs.txt",
}

OUTER_RUNS = 5
INNER_N    = 5
LENGTH     = 95


def run_eval(label: str = "current"):
    print(f"\n{'=' * 65}")
    print(f"EVAL: {label}")
    print(f"  {OUTER_RUNS} runs × {INNER_N} треков = {OUTER_RUNS * INNER_N} на модель")
    print(f"  data: {DATA_PATH}")
    print(f"{'=' * 65}")
    print(f"{'Модель':<35} {'IOU':>8} {'MSE':>8} {'ScaleCons':>10}")
    print("-" * 65)

    results = {}
    for name, path in MODELS.items():
        run_metrics = defaultdict(list)
        for _ in range(OUTER_RUNS):
            m = compare_models_avg(
                model_path=path, data_path=DATA_PATH,
                _len_=LENGTH, n=INNER_N
            )
            for k, v in m.items():
                run_metrics[k].append(v)

        results[name] = {
            "iou":   np.mean(run_metrics["iou"]),
            "mse":   np.mean(run_metrics["mse"]),
            "sc":    np.mean(run_metrics["scale_consistency"]),
        }
        r = results[name]
        print(f"{name:<35} {r['iou']:>8.4f} {r['mse']:>8.4f} {r['sc']:>10.4f}")

    print("=" * 65)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline", help="Метка запуска (для логов)")
    args = parser.parse_args()
    run_eval(args.label)
