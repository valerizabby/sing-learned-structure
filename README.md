# Генерация музыки с заданными структурными параметрами

End-to-end пайплайн авторегрессивной MIDI-генерации, управляемый двумя входными сигналами: **текстовым промтом** (характер музыки) и **сегментным планом** (музыкальная форма). Дипломная работа по направлению Machine Learning.

```
Текстовый промт          Сегментный план
      ↓                  "verse:16,chorus:16,..."
[Text2Prefix]                   ↓
      ↓                  [Seg2SSM: AffinitySSM]
  Prefix                        ↓
  (8–16 тактов MIDI)       Affinity SSM
        ↓_________________________|
              [SingLS / 3SING*]
                    ↓
              Полный MIDI-трек
```

## Результаты

| Модель | MSE ↓ | IOU ↑ | ScaleCons ↑ |
|--------|-------|-------|-------------|
| None (baseline) | 0.0645 | 0.0848 | — |
| SING (Original) | 0.0800 | 0.2986 | 0.5342 |
| LSA | 0.0568 | 0.0959 | 0.4561 |
| **3SING\*** | 0.0692 | **0.3198** | 0.5330 |

SSM-following validation: подача AffinitySSM (vs нулевая SSM) даёт ΔIOU=+0.0049, ΔMSE=−0.0057 (N=100). Механизм SSM-внимания реально управляет структурой генерации. Подробнее: [REPORT.md](REPORT.md).

## Быстрый старт

```bash
# Генерация трека с сегментным планом
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody in C major" \
    --segment "intro:8,verse:16,chorus:16,verse:16,chorus:16,outro:8" \
    --ssm_type affinity \
    --out_dir outputs/run1
```

Полное руководство и параметры: [pipeline/HOWTO.md](pipeline/HOWTO.md).

## Структура репозитория

```
SingLS/           # Модели и обучение (LSTM + SSM-внимание)
Seg2SSM/          # Построение SSM из сегментного плана (AffinitySSM)
Text2Prefix/      # Генерация MIDI-prefix по тексту (text2midi)
pipeline/         # End-to-end пайплайн и ablation study
inference/        # Оценка и визуализация
data_preparation/ # Подготовка датасетов (MAESTRO, LMD, SALAMI)
tests/            # Unit-тесты (pytest)
```

## Документация

| Документ | Содержание |
|----------|------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Архитектурные решения системы целиком |
| [Seg2SSM/ARCHITECTURE.md](Seg2SSM/ARCHITECTURE.md) | Принцип AffinitySSM, матрица A, постобработка |
| [REPORT.md](REPORT.md) | Результаты SSM-following validation и описание датасета |
| [STATUS.md](STATUS.md) | Текущее состояние: метрики, чекпоинты, итоги экспериментов |
| [pipeline/HOWTO.md](pipeline/HOWTO.md) | Руководство по запуску пайплайна и ablation study |
