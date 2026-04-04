"""
Пересчёт метрик IOU/MSE после фикса температуры (temperature=1.5 в topk_sample_one).
Запуск из корня репозитория:
    python3 -m inference.reeval
"""

import os
from inference.compare_models import compare_models_avg

DATA_PATH = "data/combined/combined_test.pt"

MODELS = {
    "SING (baseline)": "data/meta_info/trained_original_combined/model_30_epochs.txt",
    "3SING*":          "data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt",
    "LSA**":           "data/meta_info/trained_lsa_combined/model_30_epochs.txt",
}

# n — количество прогонов для усреднения, length — длина генерации в битах
N = 30
LENGTH = 95

if __name__ == "__main__":
    print(f"\nПересчёт метрик: n={N}, length={LENGTH}, data={DATA_PATH}\n")
    print(f"{'Модель':<35} {'IOU':>8} {'MSE':>8}")
    print("-" * 55)

    results = {}
    for name, path in MODELS.items():
        metrics = compare_models_avg(model_path=path, data_path=DATA_PATH, _len_=LENGTH, n=N)
        results[name] = metrics
        print(f"{name:<35} {metrics['iou']:>8.4f} {metrics['mse']:>8.4f}")

    print("\nГотово. Обнови таблицу в CLAUDE.md и в слайдах.")
