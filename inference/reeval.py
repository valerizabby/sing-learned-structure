"""
Пересчёт метрик IOU/MSE/ScaleCons после фикса температуры (temperature=1.5 в topk_sample_one).
Запуск из корня репозитория:
    python3 -m inference.reeval

Схема усреднения: compare_models_avg вызывается OUTER_RUNS раз,
каждый раз генерируя INNER_N треков. Итог усредняется по OUTER_RUNS вызовам.
Итоговая выборка: OUTER_RUNS * INNER_N = 30 * 10 = 300 треков на модель.
"""

import numpy as np
from collections import defaultdict
from inference.compare_models import compare_models_avg

DATA_PATH = "data/combined/combined_test.pt"

MODELS = {
    "SING (baseline)": "data/meta_info/trained_original_combined/model_30_epochs.txt",
    "3SING*":          "data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt",
    "LSA**":           "data/meta_info/trained_lsa_combined/model_30_epochs.txt",
}

OUTER_RUNS = 30   # сколько раз вызывать compare_models_avg
INNER_N    = 10   # сколько треков генерировать за один вызов
LENGTH     = 95

if __name__ == "__main__":
    print(f"\nПересчёт метрик: {OUTER_RUNS} runs × {INNER_N} треков = {OUTER_RUNS * INNER_N} треков на модель")
    print(f"length={LENGTH}, data={DATA_PATH}\n")
    print(f"{'Модель':<35} {'IOU':>8} {'MSE':>8} {'ScaleCons':>10}")
    print("-" * 65)

    final_results = {}

    for name, path in MODELS.items():
        run_metrics = defaultdict(list)

        for run in range(OUTER_RUNS):
            m = compare_models_avg(model_path=path, data_path=DATA_PATH, _len_=LENGTH, n=INNER_N)
            for k, v in m.items():
                run_metrics[k].append(v)

        final_results[name] = {
            "iou": np.mean(run_metrics["iou"]),
            "mse": np.mean(run_metrics["mse"]),
            "scale_consistency": np.mean(run_metrics["scale_consistency"]),
        }

    print("\n" + "=" * 65)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 65)
    print(f"{'Модель':<35} {'IOU':>8} {'MSE':>8} {'ScaleCons':>10}")
    print("-" * 65)
    for name, m in final_results.items():
        print(f"{name:<35} {m['iou']:>8.4f} {m['mse']:>8.4f} {m['scale_consistency']:>10.4f}")
    print("=" * 65)
