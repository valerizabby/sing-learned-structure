# Seg2SSM: Порядок запуска

Модуль генерирует SSM из сегментного плана (куплет/припев/...) через
**матрицу музыкального сходства (Affinity Matrix)**.

Подробная архитектура: [ARCHITECTURE.md](ARCHITECTURE.md)

Шаги 2–5 запускаются из `Seg2SSM/data_prep/`.

---

## 0. Зависимости

```bash
pip install librosa requests matplotlib
```

---

## 1. Получить аннотации SALAMI

```bash
git clone https://github.com/DDMAL/salami-data-public
```

---

## 2. Парсим аннотации

446 треков с прямыми ссылками, ручная разметка секций (verse/chorus/bridge/...).

```bash
python3 parse_annotations.py \
  --salami_dir /Users/valeriia.zaborovskaia/Desktop/diploma/salami-data-public \
  --source_filter IA \
  --class_filter all \
  --output annotations.json
```

Вывод: `annotations.json` — треки с сегментами и label_vocab.

---

## 3. Скачиваем аудио

Прямые HTTP-ссылки с archive.org, без yt-dlp.

```bash
python3 download_audio.py \
  --annotations annotations.json \
  --ia_csv /Users/valeriia.zaborovskaia/Desktop/diploma/salami-data-public/metadata/id_index_internetarchive.csv \
  --output_dir audio/
```

Ожидаемый результат: ~429 mp3, ~2–3 ГБ.

---

## 4. Извлекаем признаки (bar-chroma + SSM)

Аудио → beat tracking → бары → chroma CQT → tanh-нормализованная SSM.

```bash
python3 extract_features.py \
  --audio_dir audio/ \
  --output_dir features/
```

Вывод: `features/{song_id}.pt` — `{chroma, ssm, bar_times, n_bars}` на каждый трек.

---

## 5. Оцениваем Affinity Matrix из данных

Для каждой пары типов секций считаем среднее SSM-значение по всем трекам.
Результат — эмпирическая матрица A [8×8].

```bash
python3 estimate_affinity.py \
  --annotations annotations.json \
  --features_dir features/ \
  --output_dir ../checkpoints/
```

Вывод:
- `checkpoints/affinity_matrix.pt` — сохранённая матрица A
- `checkpoints/affinity_heatmap.png` — heatmap A + количество пар

---

## 6. Affinity SSM: использование

Модуль `affinity_ssm.py` (запускается из корня `Seg2SSM/`).

### Вариант A: теоретическая матрица (fixed)

```python
from affinity_ssm import AffinitySSM

builder = AffinitySSM.fixed()
segment_plan = [(0, 8), (1, 16), (2, 16), (1, 16), (2, 16), (5, 8)]
#               intro    verse    chorus   verse    chorus   outro
ssm = builder.build(segment_plan, ssm_size=64)  # Tensor [64, 64]
```

### Вариант B: эмпирическая матрица (из данных)

```python
from affinity_ssm import AffinitySSM

builder = AffinitySSM.from_checkpoint("checkpoints/affinity_matrix.pt")
ssm = builder.build(segment_plan, ssm_size=64)
```

### Параметры AffinitySSM

| Параметр      | Описание                                          | По умолчанию |
|---------------|---------------------------------------------------|--------------|
| `smooth_sigma`| Gaussian blur на границах блоков (0 = выкл)      | `1.5`        |
| `ssm_size`    | Размер выходной матрицы                           | `64`         |

---

## 7. Визуализация для презентации

Для `ssm_examples.png` нужен датасет — собери его один раз:

```bash
python3 build_dataset.py \
  --annotations annotations.json \
  --features_dir features/ \
  --output_dir ../../data/seg2ssm/
```

Затем генерируем три картинки для слайдов:

```bash
python3 visualize_affinity.py \
  --dataset    ../../data/seg2ssm/seg2ssm_train.pt \
  --checkpoint ../checkpoints/affinity_matrix.pt \
  --output_dir ../eval_results/affinity/
```

Вывод:
- `affinity_comparison.png` — A_fixed vs A_empirical рядом
- `ssm_examples.png` — block_ssm | affinity_fixed | affinity_emp | gt_ssm
- `segment_gallery.png` — SSM для типичных форм: Verse-Chorus, AABA, поп-форма

---

## 8. Интеграция в SingLS (следующий шаг)

`AffinitySSM` используется как "predicted SSM" в ablation study:

| Режим         | Источник SSM                                     |
|---------------|--------------------------------------------------|
| No SSM        | zeros                                            |
| GT SSM        | реальный chroma SSM из piano roll                |
| **Affinity SSM** | `AffinitySSM.fixed().build(segment_plan)`     |
| Random SSM    | случайный                                        |

---

## Структура файлов

```
Seg2SSM/
├── ARCHITECTURE.md            ← архитектура и обоснование подхода
├── Plan.md                    ← этот файл
├── affinity_ssm.py            ← основной модуль (AffinitySSM)
├── checkpoints/
│   ├── affinity_matrix.pt     ← эмпирическая матрица A
│   └── affinity_heatmap.png   ← визуализация матрицы
├── eval_results/affinity/     ← картинки для презентации
└── data_prep/
    ├── parse_annotations.py   ← шаг 2
    ├── download_audio.py      ← шаг 3
    ├── extract_features.py    ← шаг 4
    ├── estimate_affinity.py   ← шаг 5
    ├── build_dataset.py       ← вспомогательный (для визуализации)
    ├── visualize.py           ← проверка датасета
    └── visualize_affinity.py  ← шаг 7 (слайды)
```
