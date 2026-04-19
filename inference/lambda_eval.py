"""
Оценка моделей из lambda_search по метрикам IOU/MSE/ScaleCons.
Запуск из корня репозитория:
    python3 -m inference.lambda_eval

Схема: OUTER_RUNS × INNER_N треков на модель (как в reeval.py).
"""

import numpy as np
from collections import defaultdict
from inference.compare_models import compare_models_avg

DATA_PATH = "data/combined/combined_test.pt"
LAMBDA_BASE = "experiments/sing-results/meta_info/lambda_search"

MODELS = {
    "SING λ=1":  f"{LAMBDA_BASE}/sing_lambda_1/model_final.pt",
    "SING λ=5":  f"{LAMBDA_BASE}/sing_lambda_5/model_final.pt",
    "SING λ=10": f"{LAMBDA_BASE}/sing_lambda_10/model_final.pt",
    "SING λ=20": f"{LAMBDA_BASE}/sing_lambda_20/model_final.pt",
}

OUTER_RUNS = 30
INNER_N    = 10
LENGTH     = 95

if __name__ == "__main__":
    print(f"\nLambda search eval: {OUTER_RUNS} runs × {INNER_N} = {OUTER_RUNS * INNER_N} треков на модель")
    print(f"length={LENGTH}, data={DATA_PATH}\n")

    final_results = {}

    for name, path in MODELS.items():
        print(f"Evaluating {name} ...")
        run_metrics = defaultdict(list)

        for run in range(OUTER_RUNS):
            m = compare_models_avg(model_path=path, data_path=DATA_PATH, _len_=LENGTH, n=INNER_N)
            for k, v in m.items():
                run_metrics[k].append(v)
            if (run + 1) % 5 == 0:
                print(f"  run {run + 1}/{OUTER_RUNS}")

        final_results[name] = {
            "iou":               np.mean(run_metrics["iou"]),
            "mse":               np.mean(run_metrics["mse"]),
            "scale_consistency": np.mean(run_metrics["scale_consistency"]),
        }
        print(f"  -> IOU={final_results[name]['iou']:.4f}  MSE={final_results[name]['mse']:.4f}  ScaleCons={final_results[name]['scale_consistency']:.4f}")

    print("\n" + "=" * 65)
    print("LAMBDA SEARCH — ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 65)
    print(f"{'Модель':<20} {'IOU':>8} {'MSE':>8} {'ScaleCons':>12}")
    print("-" * 65)
    for name, m in final_results.items():
        print(f"{name:<20} {m['iou']:>8.4f} {m['mse']:>8.4f} {m['scale_consistency']:>12.4f}")
    print("=" * 65)

    best_iou = max(final_results, key=lambda n: final_results[n]["iou"])
    best_mse = min(final_results, key=lambda n: final_results[n]["mse"])
    print(f"\nЛучший по IOU:  {best_iou}  ({final_results[best_iou]['iou']:.4f})")
    print(f"Лучший по MSE:  {best_mse}  ({final_results[best_mse]['mse']:.4f})")
