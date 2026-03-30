# CLAUDE.md — sing-learned-structure

Дипломная работа: **"Генерация музыки с заданными структурными параметрами"**

> Справочная база знаний по MIR, доменам и моделям: [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md)

---

## Контекст проекта

Проект расширяет [SING](https://github.com/sophia-hager/SING) (Similarity Incentivized Neural Generator) — систему авторегрессивной генерации MIDI-музыки, в которой SSM (self-similarity matrix) используется как механизм структурного внимания.

Цель дипломной работы — построить **end-to-end пайплайн генерации музыки**, управляемый двумя входными сигналами:
1. **Сегментный план** (список разделов: куплет, припев и т.д.) → SSM
2. **Текстовый промт** (описание композиции) → короткий музыкальный отрезок (prefix)

Эти два компонента подаются на вход лучшей модели SingLS, которая генерирует полный трек.

---

## Планируемая end-to-end архитектура

```
Текстовый промт          Сегментный план
      ↓                  (куплет/припев/бридж...)
[Text → Prefix]                 ↓
(готовое решение,        [Seg2SSM: AffinitySSM]
 совместимое с MIDI)     (аналитически, без обучения)
      ↓                         ↓
  Prefix                  Affinity SSM
  (8–16 тактов MIDI)      ↓_________________________|
             [SingLS]
         (лучшая конфигурация)
                ↓
          Полный MIDI-трек
```

**Подзадача 1 — Segment → SSM (реализовано):**
- Датасет: SALAMI — 422 трека с разметкой секций
- Подход: матрица музыкального сходства A [8×8]
  - `A[label_i, label_j]` = сходство между барами типа `label_i` и `label_j`
  - `A_fixed`: задана из теории музыки (verse↔chorus=0.7, bridge контрастирует)
  - `A_empirical`: оценена из данных как среднее SSM по парам меток
- Построение: `segment_plan → bar_labels → A[bar_labels][:, bar_labels] → SSM`
- Опционально: Gaussian smoothing на границах блоков
- Модуль: `Seg2SSM/affinity_ssm.py`, класс `AffinitySSM`
- Детали: `Seg2SSM/ARCHITECTURE.md`

**Подзадача 2 — Text → Prefix:**
- Использовать готовую модель, генерирующую MIDI по тексту (8–16 тактов)
- Требование: совместимость с доменом piano roll `[T, 128]`

**Обязательные абляции при интеграции:**
1. No SSM (baseline)
2. GT SSM (верхняя граница)
3. Affinity SSM — `AffinitySSM.fixed().build(segment_plan)` (целевой режим)
4. Random SSM (нижняя граница)

---

## Текущее состояние модели

### Метрики (датасет combined, 2600 треков)

| Модель | MSE ↓ | IOU ↑ |
|--------|-------|-------|
| None (baseline) | 0.0645 | 0.0848 |
| SING (Original) | 0.1628 | 0.2006 |
| LSA | 0.0682 | 0.1450 |
| LSA_SB | 0.0583 | 0.0767 |
| Transformer + beta=0.03 | 0.1007 | 0.2751 |

Лучшая модель по IOU — **HierarchicalGenerator с beta=0.03 и структурой как регуляризацией** (`trained_transformer_original_less_struct_combined`).
Лучшая по MSE — **LSA_SB**.

---

## Структура репозитория

### Активный код (SingLS/)

```
SingLS/
├── model/
│   ├── model.py                  # MusicGenerator — LSTM + выбор механизма внимания
│   ├── original_attention.py     # Оригинальный SING attention (sparsemax + SSM)
│   ├── lsa.py                    # LSA — QKV + мультипликативное маскирование SSM
│   ├── lsaSB.py                  # LSA_SB — аддитивный logit-бонус: scores += β·SSM
│   ├── structure_transformer.py  # StructureTransformer + StructureModel (глобальный план)
│   ├── hierarchical_generator.py # HierarchicalGenerator = MusicGenerator + StructureModel
│   └── utils.py                  # build_ssm_batch, freeze/unfreeze structure
├── trainer/
│   ├── train.py                  # ModelTrainer: цикл обучения, custom_loss, generate
│   └── data_utils.py             # Батчинг, SSM-вычисление, top-k сэмплирование
├── config/
│   └── config.py                 # DEVICE, hidden_size, AttentionType enum
└── main.py                       # Точка входа: выбор модели, обучение
```

### Инференс и оценка

```
inference/
├── main.py           # Сравнение моделей по IOU/MSE на combined_test
├── artifacts.py      # Визуализации: SSM, piano roll, экспорт в MIDI
└── compare_models.py # iou_ssm(), mse_ssm(), piano_roll_to_midi()
```

### Подготовка данных

```
data_preparation/
├── LMD/
│   ├── build_LMD.py          # LMD → MAESTRO-совместимый формат
│   ├── data_preprocessing.py # MIDI → piano roll (tempo-adjusted, binarized)
│   └── build_BINS.py         # train/val/test split
└── union.py                  # LMD + MAESTRO → combined_*.pt
```

### Seg2SSM — генерация SSM из сегментного плана

```
Seg2SSM/
├── ARCHITECTURE.md            # архитектура и обоснование подхода
├── Plan.md                    # порядок запуска
├── affinity_ssm.py            # AffinitySSM: segment_plan → SSM [64×64]
├── checkpoints/
│   ├── affinity_matrix.pt     # эмпирическая матрица A (422 трека SALAMI)
│   └── affinity_heatmap.png   # визуализация матрицы
├── eval_results/affinity/     # картинки для презентации
└── data_prep/
    ├── parse_annotations.py   # SALAMI → annotations.json
    ├── download_audio.py      # скачивание mp3 с archive.org
    ├── extract_features.py    # audio → bar-chroma + tanh-SSM
    ├── estimate_affinity.py   # оценка A из данных
    ├── build_dataset.py       # сборка датасета (для визуализации)
    ├── visualize.py           # проверка датасета
    └── visualize_affinity.py  # слайды: A_fixed vs A_emp, примеры SSM
```

### Legacy-файлы (ориентир, не трогать)

```
att-lstm.py           # Legacy: оригинальный скрипт обучения SING
model-selection.py    # Legacy: поиск лучшего чекпоинта по val loss
test.py               # Legacy: вспомогательный скрипт тестирования
data_processing.ipynb # Legacy: исходная обработка MAESTRO (Colab-ноутбук)
gen-example.ipynb     # Legacy: генерация примеров из оригинального SING
test-sim.ipynb        # Legacy: анализ SSM
```

---

## Ключевые концепции

### SSM (Self-Similarity Matrix)
- Строится по **chroma-признакам**: piano roll `[T, 128]` → chroma `[T, 12]` → cosine similarity `[T, T]`
- Отражает структурное сходство моментов времени
- Используется как attention bias при генерации

### Piano Roll
- Бинарная матрица `[T, 128]`, где T = количество битов/тактов, 128 = MIDI-ноты
- Семплируется с темпо-адаптивной частотой: `fs = tempo/60`
- Диапазон валидных нот: 20–108

### AttentionType
```python
class AttentionType(Enum):
    NONE     # Без attention (baseline)
    ORIGINAL # SING: sparsemax по SSM
    LSA      # QKV + SSM как мультипликативная маска
    LSA_SB   # QKV + SSM как аддитивный логит: scores += β·SSM
```

### Loss-функция
```
total_loss = BCE(output, target) + SSM_L2_loss + β_struct * structure_loss
```
- `BCE` — бинарная кросс-энтропия по piano roll
- `SSM_L2_loss` — L2 между chroma-SSM генерации и оригинала
- `structure_loss` — контрастивный loss на структурных эмбеддингах (опционально)
- `β_struct = 0.03`

---

## Датасеты

| Датасет         | Треков     | Формат                          |
|-----------------|------------|---------------------------------|
| MAESTRO         | ~1300      | piano roll `.pt`                |
| LMD (aligned)   | ~1300      | piano roll `.pt`                |
| combined        | ~2600      | `combined_train/val/test.pt`    |

Формат элемента: `(tensor[T, 128], tempo, num_beats)`

---

## Правила разработки

1. **Не изменять legacy-файлы** — они нужны как эталон логики оригинального SING.
2. **Новый функционал** — только в `SingLS/` или новых директориях (например, `Seg2SSM/`).
3. **Эксперименты** фиксируются в `SingLS/README.md` с метриками IOU и MSE.
4. **Чекпоинты** именуются по схеме: `trained_{модель}_{данные}_{особенность}`.
5. **Конфигурация** — через `SingLS/config/config.py`, не хардкодить в скриптах.
6. При добавлении новых модулей соблюдать интерфейс: `forward(x, ssm, hidden) → (output, hidden)`.
