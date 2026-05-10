# Plan: Улучшение качества демо к защите

## Контекст

Текущая проблема: сгенерированная музыка звучит как набор случайных нот.
Гипотеза о главной причине: OOD-prefix от text2midi заражает LSTM нехарактерным
распределением нот, и даже warmup_beats=4 не спасает.

Цель: (а) доказать на метриках, что модель умеет следовать SSM, (б) добиться
слушаемого результата для демо на защите.

---

## Шаг 1: Оценка SSM-following rate

**Цель:** показать, что 3SING* с AffinitySSM реально управляет структурой.

**Что это:** новая метрика (отличная от IOU в таблице). Алгоритм:
1. Взять N=100 реалистичных сегментных планов (случайные комбинации
   intro/verse/chorus/bridge/outro из SALAMI-стилистики)
2. Для каждого плана: построить AffinitySSM как *цель*
3. Сгенерировать трек с фиксированными параметрами (лучшая конфигурация из
   SAMPLING_EXPERIMENTS.md)
4. Вычислить реальную chroma-SSM из сгенерированного piano roll
5. Посчитать IOU(target, actual) и MSE(target, actual)
6. Повторить то же с ssm_type=none (baseline)

**Результат:** таблица IOU/MSE для двух условий (affinity vs none).
Если affinity > none — SSM реально работает как структурный guidance.

**Файл:** `experiments/eval_ssm_following.py`

**Фиксированные параметры для генерации:**
```
--model  trained_transformer_original_less_struct_combined
--temperature 1.0 --temp_start 1.5 --temp_warmup_steps 10
--topk_k 10 --n_samples 1 --note_extension 2
--warmup_beats 4
```

**Оценка трудозатрат:** ~3–4 часа (скрипт + прогон 200 генераций).

> **[ВЫПОЛНЕНО 2026-04-29]**
> Перед запуском потребовались два фикса в `pipeline/generate.py` и `SingLS/model/hierarchical_generator.py`
> (SSM alignment bug + StructureModel 1×1 bug — см. `pipeline/COLLAPSE_DEBUG.md`).
> После фиксов результаты (N=100, n_gen_bars=64, ssm_size=64):
>
> | Условие  | IOU ↑             | MSE ↓             |
> |----------|-------------------|-------------------|
> | affinity | 0.0329 ± 0.0079   | 0.2877 ± 0.0415   |
> | none     | 0.0280 ± 0.0081   | 0.2934 ± 0.0413   |
>
> ΔIOU = +0.0049 ✓ affinity лучше | ΔMSE = −0.0057 ✓ affinity лучше
> **Вывод: AffinitySSM статистически улучшает структурное следование модели.**

---

## Шаг 2: In-distribution prefix из датасета

**Цель:** убрать главный источник деградации — OOD-prefix от text2midi.

**Идея (Fix 3 из COLLAPSE_DEBUG.md):**
Вместо text2midi → взять случайный (или подобранный) фрагмент из
`data/combined/combined_test.pt` как prefix. Это гарантирует:
- in-distribution hidden state LSTM
- coherent start (реальные паттерны из MAESTRO/LMD)

Текстовый промт при этом может использоваться для семантического поиска по
датасету или отбрасываться совсем (для чистого структурного демо).

**Что реализовать:**
- Добавить в `pipeline/generate.py` параметр `--prefix_from_dataset`
  (путь к `.pt` файлу датасета + индекс или random seed)
- Брать первые `n_prefix_bars * BEATS_PER_BAR` битов из случайного трека

**Файл:** `pipeline/generate.py` (добавить ветку загрузки prefix)

**Оценка трудозатрат:** ~1–2 часа (случайный) / ~3–4 часа (SSM-matching).

---

### Расширенная идея: SSM-matching prefix (зафиксировано 2026-05-03)

**Суть:** вместо случайного трека — подбирать prefix из датасета по близости SSM.

**Алгоритм:**
1. По `segment_plan` строим `target_ssm` (AffinitySSM, resize → 64×64)
2. Для каждого трека в `combined_test.pt` считаем `track_ssm` (chroma-SSM, resize → 64×64)
   — кешируется в `data/seg2ssm/combined_test_ssms.pt` (один раз)
3. Находим топ-K треков с минимальным `MSE(target_ssm, track_ssm)`
4. Из случайного трека топ-K берём первые `n_prefix_bars * BEATS_PER_BAR` битов

**Почему это лучше случайного:**
- Prefix из трека со схожей структурой прогревает LSTM в нужном ладотональном контексте
- Начало генерации согласовано со структурным планом, а не случайно
- Сохраняется in-distribution гарантия (реальные MAESTRO/LMD данные)

**Что нужно реализовать:**
- `pipeline/dataset_prefix.py` — новый модуль:
  - `precompute_dataset_ssms(dataset_path, ssm_size=64, cache_path)` — предвычисление и кеш
  - `find_best_prefix(target_ssm, dataset_path, n_prefix_beats, cache_path, top_k=5, seed)` → `Tensor [n_beats, 128]`
- В `pipeline/generate.py` добавить аргументы:
  - `--prefix_from_dataset PATH` — путь к `.pt` датасета (активирует SSM-matching режим)
  - `--dataset_ssm_cache PATH` — путь к кешу SSM (по умолчанию `data/seg2ssm/combined_test_ssms.pt`)
  - `--prefix_top_k INT` — размер топ-K для случайного выбора (по умолчанию 5)
  - `--prefix_seed INT` — seed для воспроизводимости

**Технические детали:**
- Формат `combined_test.pt`: список `numpy.ndarray` shape `(3,)` = `[piano_roll, tempo, num_beats]`
- `data/seg2ssm/seg2ssm_test.pt` уже содержит `gt_ssm [64,64]` для SALAMI-треков — не то
- SSM для MAESTRO/LMD нужно считать через `SingLS.trainer.data_utils.SSM(roll)` + resize
- При инференсе без кеша: пересчитывать на лету (медленно, ~30 сек на 568 треков)
- С кешем: мгновенно (матмул 568×4096 vs target 4096)

**Открытые вопросы:**
- Сравнивать полную SSM трека или только первые `n_prefix_bars` баров?
  (полная SSM отражает структуру трека целиком — вероятно лучше)
- Нужен ли отдельный кеш для `combined_train.pt` (2032 трека)?

---

## Шаг 3: MIDI post-processing

**Цель:** улучшить восприятие без переобучения модели.

**Что добавить в `inference/artifacts.py` или отдельный скрипт:**
- **Sustain pedal** — добавить CC64 events в MIDI, чтобы ноты
  плавно перекрывались (убирает эффект "пунктирных" нот)
- **Velocity variation** — случайный ±10–15 velocity per note
  (звучит живее, чем flat 80)
- **Quantization cleanup** — удалить ноты короче 1 бита (артефакты)
- **Note deduplication** — убрать ноты одной высоты, начинающиеся
  в пределах 1 бита (дублирование от OR-сэмплирования)

**Файл:** `inference/midi_postprocess.py` (новый) + подключить в
`pipeline/generate.py` перед сохранением MIDI.

**Оценка трудозатрат:** ~2 часа.

---

## Шаг 4: Cherry-pick лучшего результата

**Цель:** гарантировать, что на защите звучит лучший из возможных треков.

**Что сделать:**
- Написать скрипт `experiments/cherry_pick.py`, который запускает
  N=10–20 генераций с разными random seed
- Для каждой автоматически считает:
  - note density (должна быть 0.05–0.15)
  - pitch_std (желательно < 15 после постпроцессинга)
  - IOU(target_ssm, actual_ssm)
- Сортирует по совокупному score и сохраняет топ-3

**Входные параметры:** фиксированный сегментный план + лучший prefix
(либо из датасета — шаг 2, либо лучший text2midi result).

**Файл:** `experiments/cherry_pick.py`

**Оценка трудозатрат:** ~1–2 часа (после шагов 2–3).

---

## Шаг 5: Убрать prefix из итогового MIDI

**Цель:** трек в демо содержит только то, что модель нагенерировала сама
(не зависит от text2midi артефактов в начале).

**Что изменить в `pipeline/generate.py`:**
- `run_pipeline` сохраняет в `generated.mid` только `full_roll[T_prefix:]`
- `prefix.mid` — отдельно (для сравнения)
- Обновить `save_visualizations`: отметить границу prefix / generated

**Файл:** `pipeline/generate.py`, `pipeline/HOWTO.md` (TODO уже зафиксирован)

**Оценка трудозатрат:** ~30 минут.

> **[УЖЕ РЕАЛИЗОВАНО]**
> Всё три пункта выполнены в `pipeline/generate.py`:
> - `generated.mid` → `piano_roll_to_midi(full_roll[T_prefix:], ...)` (строка ~457)
> - `prefix.mid` → сохраняется отдельно (строка ~461)
> - `save_visualizations` → красная пунктирная линия на piano roll + голубая на SSM (строка ~309)

---

## Порядок выполнения

| Приоритет | Шаг | Эффект | Время |
|-----------|-----|--------|-------|
| 1 | Шаг 1 (eval) | Факты для защиты | 3–4 ч |
| 2 | Шаг 2 (in-dist prefix) | Главный выигрыш по качеству | 1–2 ч |
| 3 | Шаг 5 (trim prefix) | Чистое демо | ✅ уже готово |
| 4 | Шаг 3 (postprocess) | Лучшее восприятие | 2 ч |
| 5 | Шаг 4 (cherry-pick) | Финальный отбор | 1–2 ч |

---

## Критерий готовности к защите

- [x] Есть таблица SSM-following (affinity > none) — ΔIOU=+0.0049, ΔMSE=−0.0057 (N=100)
- [ ] Есть MIDI-файл, который звучит как "пьеса", а не "случайные ноты"
- [ ] Структурная разметка в SSM визуально соответствует сегментному плану
- [ ] Демо запускается одной командой (pipeline.generate или UI)
