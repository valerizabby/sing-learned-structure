# Seg2SSM: Affinity-based SSM Generation

## Идея

SSM строится не нейросетью, а через явную матрицу музыкального сходства между
типами секций (Affinity Matrix). Это даёт интерпретируемый, визуально понятный
модуль, который можно объяснить на одном слайде.

```
segment_plan                       Affinity Matrix A (7×7)
[(verse,16),(chorus,16),...]
        │                        intro  verse chorus bridge instr outro other
        ▼                 intro [  1.0   0.6   0.4    0.2   0.5   0.6   0.2 ]
  bar_labels [T]          verse [  0.6   1.0   0.7    0.3   0.4   0.3   0.2 ]
  [1,1,...,2,2,...]      chorus [  0.4   0.7   1.0    0.4   0.5   0.5   0.2 ]
        │                bridge [  0.2   0.3   0.4    1.0   0.3   0.2   0.2 ]
        ▼                 instr [  0.5   0.4   0.5    0.3   1.0   0.4   0.2 ]
  SSM[i,j] = A[lbl_i, lbl_j]    outro [  0.6   0.3   0.5    0.2   0.4   1.0   0.2 ]
        │                 other [  0.2   0.2   0.2    0.2   0.2   0.2   1.0 ]
        ▼
  resize → SSM [64×64]
```

## Два варианта матрицы A

### A_fixed
Задаётся вручную на основе теории музыки.
Прозрачно, без обучения, воспроизводимо.
Используется как baseline и для интерпретации.

Принципы заполнения:
- Диагональ = 1.0 (секция идентична себе)
- verse ↔ chorus: высокое (0.7) — оба несут основной материал
- intro ↔ outro: среднее (0.6) — обрамляют трек, часто схожи
- bridge: низкое со всеми (0.2–0.4) — контрастная секция
- instr: среднее (0.4–0.5) — инструментальный вариант verse/chorus

### A_empirical
Оценивается из данных без backprop:

```
A[a, b] = mean( GT_SSM[i,j] )
          для всех пар (i,j) где label(bar_i)=a, label(bar_j)=b
```

Источник данных: SALAMI (annotations.json + features/).
Матрица A сама по себе является результатом — её heatmap показывает
что "узнали" из реальных треков о музыкальных отношениях.

## Построение SSM

```python
def build_affinity_ssm(segment_plan, A, T, ssm_size=64, smooth_sigma=1.5,
                       noise_std=0.04, rescale=True):
    bar_labels = build_bar_labels(segment_plan, T)  # [T]
    ssm = A[bar_labels][:, bar_labels]              # [T, T]
    if smooth_sigma > 0:
        ssm = gaussian_blur(ssm, sigma=smooth_sigma) # сглаживаем границы блоков
    if noise_std > 0:
        noise = randn(T, T) * noise_std
        ssm += symmetrize(noise)                    # внутриблочная вариация
        ssm.diagonal = 1.0; ssm.clamp(0, 1)
    if rescale:
        ssm = linear_rescale(ssm,                   # под распределение chroma-SSM
                             target_mean=0.50,
                             target_std=0.13)
    return resize_ssm(ssm, ssm_size)                 # [ssm_size, ssm_size]
```

### Зачем постобработка?

Модель **обучена на chroma-SSM**: cosine similarity нормированных chroma-векторов.
Типичное распределение: mean≈0.50, std≈0.13, плавные переходы.

AffinitySSM без постобработки (`rescale=False, noise_std=0`):
- mean≈0.70, std≈0.22 — другое распределение
- Ступенчатые границы блоков — нереалистичная текстура
- Только 3–4 дискретных значения во всей матрице

После постобработки:
- mean≈0.50, std≈0.13 ✓ — совпадает с обучающим распределением
- Сглаженные границы (Gaussian blur σ=1.5 ≈ 1 такт)
- Вариация внутри блоков (noise_std=0.04) — имитирует естественную вариабельность chroma

Структурный паттерн (verse≈chorus, bridge контрастирует) **сохраняется** — 
rescale — это линейное масштабирование, не уничтожающее порядок значений.



## План реализации

### Шаг 1: estimate_affinity.py
- Загружаем annotations.json + features/ (bar chroma + bar_times)
- Для каждого трека: выравниваем сегменты → bar_labels
- Считаем среднее tanh-нормализованное SSM[i,j] по всем парам (label_a, label_b)
- Сохраняем A как affinity_matrix.pt
- Визуализируем как heatmap (affinity_heatmap.png)

### Шаг 2: affinity_ssm.py
- Класс AffinitySSM: хранит A, строит SSM для любого segment_plan
- Два режима: fixed / empirical
- Поддержка Gaussian smoothing
- Метод visualize() для слайдов

### Шаг 3: visualize_affinity.py
- Для N примеров показываем рядом:
  block_ssm | affinity_ssm | gt_ssm
- Отдельный heatmap матрицы A с подписями секций

### Шаг 4: Интеграция в SingLS (ablation)
- affinity_ssm как "predicted SSM" в ablation study
- Сравнение: GT SSM / affinity SSM / block SSM / random / none

## Что показываем на защите

Один слайд — три картинки:

  [Heatmap A]        [segment_plan → SSM]       [GT SSM из MIDI]
  "что выучили"      "что предсказываем"         "upper bound"

Нарратив:
  "Мы описываем структурные отношения между секциями компактной матрицей
   музыкального сходства. Матрица интерпретируема: куплет близок к припеву,
   бридж контрастирует с обоими. Построенная SSM используется как структурный
   prior для авторегрессивной MIDI-генерации."

## Файловая структура

```
Seg2SSM/
├── ARCHITECTURE.md          ← этот файл
├── affinity_ssm.py          ← основной модуль (AffinitySSM)
├── data_prep/
│   ├── estimate_affinity.py ← оценка A из данных
│   └── visualize_affinity.py← визуализация A + примеры SSM
└── checkpoints/
    └── affinity_matrix.pt   ← сохранённая матрица A
```
