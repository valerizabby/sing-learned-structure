# Pipeline — инструкция по запуску

Все команды запускаются из **корня репозитория** (`sing-learned-structure/`).

---

## Доступные модели

Чекпоинты лежат в `data/meta_info/` с расширением `.txt` — это обычные
PyTorch-файлы (`torch.save`), расширение не влияет на загрузку.

| Директория | Описание | IOU ↑ | MSE ↓ | ScaleCons ↑ |
|------------|----------|-------|-------|-------------|
| `trained_transformer_original_less_struct_combined` | **3SING*** — HierarchicalGenerator, β=0.03 | 0.3198 | 0.0692 | 0.5330 |
| `trained_lsa_combined` | LSA attention | 0.0959 | 0.0568 | 0.4561 |
| `trained_original_combined` | SING original attention | 0.2986 | 0.0800 | 0.5342 |

Путь к чекпоинту: `data/meta_info/<название>/model_30_epochs.txt`

---

## Быстрый старт

### Генерация одного трека

```bash
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody in C major, slow tempo" \
    --segment "intro:4,verse:8,chorus:8,verse:8,chorus:8,outro:4" \
    --tempo 90 \
    --n_gen_bars 36 \
    --ssm_type affinity \
    --warmup_beats 4 \
    --out_dir outputs/my_run
```

> **Важно:** `--segment` описывает полную структуру трека (prefix + generated).
> Ориентир для суммы баров: `8 + n_gen_bars` (8 — бюджет prefix).
> Точный T_prefix определяется text2midi автоматически; SSM масштабируется к реальному T_total.
> В примере: 4+8+8+8+8+4 = 40 ≈ 8+32. ✓

**Результаты** сохранятся в `outputs/my_run/`:
- `generated.mid` — полный MIDI-трек (prefix + generated)
- `prefix.mid` — только prefix (для сравнения)
- `prefix_roll.pt` — сохранённый prefix (для переиспользования)
- `pipeline_overview.png` — 3 панели: prefix piano roll | full piano roll | SSM heatmap

---

### Повторная генерация с тем же prefix

Prefix от text2midi генерируется долго (~1–2 мин). Чтобы не ждать повторно:

```bash
# Первый запуск — prefix_roll.pt сохраняется автоматически
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody in C major" \
    --segment "intro:4,verse:8,chorus:8,verse:8,chorus:8,outro:4" \
    --n_gen_bars 36 \
    --out_dir outputs/run1

# Повторные запуски — загружает сохранённый prefix, пропускает text2midi
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody in C major" \
    --segment "intro:4,verse:8,chorus:8,verse:8,chorus:8,outro:4" \
    --n_gen_bars 36 \
    --prefix_pt outputs/run1/prefix_roll.pt \
    --temperature 1.8 \
    --out_dir outputs/run1_hot
```

---

### Ablation study (все 4 типа SSM за один запуск)

```bash
python -m pipeline.ablation \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody in C major, slow tempo" \
    --segment "intro:4,verse:8,chorus:8,verse:8,chorus:8,outro:4" \
    --tempo 90 \
    --n_gen_bars 36 \
    --out_dir outputs/ablation
```

**Результаты** сохранятся в `outputs/ablation/`:
```
ablation/
├── none/generated.mid
├── random/generated.mid
├── affinity/generated.mid
├── empirical/generated.mid      # только если есть affinity_matrix.pt
├── prefix.mid
├── ablation_overview.png        # сетка 2×4: piano rolls + SSM
└── metrics.txt                  # IOU и MSE для каждого условия
```

---

## Параметры

### `pipeline.generate`

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--model` | — | **Обязательный.** Путь к чекпоинту SingLS (`.txt` или `.pt`) |
| `--prompt` | — | **Обязательный.** Текстовый промт для text2midi |
| `--segment` | — | **Обязательный.** Сегментный план (см. ниже) |
| `--tempo` | `120.0` | Темп в BPM |
| `--n_gen_bars` | `64` | Длина генерируемой части в барах |
| `--ssm_type` | `affinity` | Тип SSM: `affinity` / `empirical` / `random` / `none` |
| `--empirical_ckpt` | `Seg2SSM/checkpoints/affinity_matrix.pt` | Матрица A для `empirical` |
| `--warmup_beats` | `4` | Сколько последних битов prefix пускать через LSTM (см. ниже) |
| `--prefix_in_sequence` | `False` (флаг) | Включить OOD-фреймы prefix в attention context (не рекомендуется) |
| `--temperature` | `1.5` | Температура сэмплирования |
| `--prefix_pt` | `None` | Путь к сохранённому `prefix_roll.pt` (пропустить text2midi) |
| `--out_dir` | `outputs` | Директория для сохранения результатов |

### `pipeline.ablation`

Те же параметры, кроме `--ssm_type` (ablation запускает все типы автоматически).

---

## Сегментный план

Формат: `"label:n_bars,label:n_bars,..."`

Доступные метки:

| Метка | ID | Описание |
|-------|----|----------|
| `intro` | 0 | Вступление |
| `verse` | 1 | Куплет |
| `chorus` | 2 | Припев |
| `bridge` | 3 | Бридж / переход |
| `instr` | 4 | Инструментальный раздел |
| `outro` | 5 | Завершение |
| `other` | 6 | Прочее |

**Примеры:**
```bash
# Классическая поп-структура (n_prefix_bars=4, n_gen_bars=40)
--segment "intro:4,verse:8,chorus:8,verse:8,chorus:8,bridge:4,chorus:8,outro:4"

# Короткий тест для быстрой проверки пайплайна (n_prefix_bars=4, n_gen_bars=8)
--segment "verse:4,chorus:4,verse:4"

# С инструментальным разделом (n_prefix_bars=4, n_gen_bars=36)
--segment "intro:4,verse:8,chorus:8,instr:8,chorus:8,outro:4"
```

> **Правило:** `sum(n_bars для всех секций) == n_prefix_bars + n_gen_bars`

---

## Параметр `--warmup_beats`

Модель SingLS обучалась на данных MAESTRO/LMD. Prefix от text2midi —
OOD (out-of-distribution): другой домен, другое распределение нот.

Если пропустить весь prefix через LSTM, hidden state уходит в патологическую
область → генерация коллапсирует к 2–3 повторяющимся нотам.

**Решение:** через LSTM пропускаются только последние `warmup_beats` битов prefix.
При этом OOD-фреймы **не попадают в sequence** (см. `--prefix_in_sequence`),
так что attention видит только сгенерированные in-distribution ноты.

| `--warmup_beats` | Эффект |
|-----------------|--------|
| `0` | hidden state = нули; максимально хаотично, но без коллапса |
| `4` | **Рекомендуемый дефолт** — минимальный контекст, стабильная генерация |
| `≥ 10` | Высокий риск коллапса при OOD prefix |

---

## Параметр `--prefix_in_sequence`

По умолчанию OOD-фреймы prefix **не попадают** в `sequence` — attention-механизм
видит только сгенерированные in-distribution ноты (Fix 2 из COLLAPSE_DEBUG.md).

```bash
# Рекомендуемый режим (по умолчанию, флаг НЕ указывается):
# prefix только прогревает LSTM, attention его не видит
python -m pipeline.generate ...

# Старое поведение (OOD виден attention — высокий риск коллапса):
python -m pipeline.generate ... --prefix_in_sequence
```

---

## Типы SSM

| Тип | Назначение | Описание |
|-----|-----------|----------|
| `affinity` | **Целевой режим** | Теоретическая матрица A_fixed, без обучения |
| `empirical` | Верхний ориентир | Матрица A, оценённая из 422 треков SALAMI |
| `random` | Нижняя граница | Случайная симметричная SSM |
| `none` | Baseline | Нулевая SSM (модель игнорирует структуру) |

---

## Использование эмпирической матрицы

Чекпоинт `Seg2SSM/checkpoints/affinity_matrix.pt` уже лежит в репозитории.

```bash
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "upbeat jazz piano, 120 bpm" \
    --segment "intro:4,verse:8,chorus:8,verse:8,chorus:8,outro:4" \
    --tempo 120 \
    --n_prefix_bars 4 --n_gen_bars 36 \
    --ssm_type empirical \
    --empirical_ckpt Seg2SSM/checkpoints/affinity_matrix.pt \
    --out_dir outputs/empirical_run
```

---

## Установка зависимостей

```bash
# Основные зависимости пайплайна
pip install torch pretty_midi matplotlib

# text2midi (прямая MIDI-генерация — основной бэкенд, рекомендуется)
pip install transformers huggingface_hub "miditok>=3.0,<4.0" symusic jsonlines

# Sparsemax (для SING original attention)
pip install sparsemax
```

При **первом запуске** text2midi автоматически скачает модель (~1 GB) с HuggingFace.
Последующие запуски используют локальный кэш (`~/.cache/huggingface/hub/`).

Если text2midi недоступен, пайплайн автоматически переключится на
MusicGen+BasicPitch (аудио-бэкенд):
```bash
pip install audiocraft basic-pitch librosa soundfile
```

---

## Устранение проблем

**Генерация коллапсирует к 2–3 повторяющимся нотам:**
```bash
# Убедись что --prefix_in_sequence НЕ задан (дефолт)
# Уменьши warmup_beats:
--warmup_beats 4
# Или полностью выключи warmup для чистого старта:
--warmup_beats 0
```

**Prefix короче запрошенного (`[!] text2midi сгенерировал только N/M битов` в логах):**
- Нормальное поведение: text2midi может завершиться раньше запрошенных `n_prefix_bars`
  (ранний EOS). Пайплайн автоматически обрезает `prefix_roll` до реального контента
  и сдвигает красную линию влево. SSM и длина генерации пересчитываются корректно.
- Чтобы получить более длинный prefix — запроси больше баров: `--n_prefix_bars 12`

**Prefix почти пустой (`Prefix density: 0.0000` в логах):**
- text2midi иногда генерирует очень тихую/медленную музыку. Добавь BPM явно:
  `--prompt "gentle piano melody, 90 BPM"`
- Попробуй более конкретный промт с описанием инструмента и стиля

**`max_notes = 0` / генерация завершается сразу:**
Сумма баров в `--segment` не совпадает с `n_prefix_bars + n_gen_bars`.
Проверь: `sum(все n_bars в --segment) == --n_prefix_bars + --n_gen_bars`

**text2midi: `ValueError: Unrecognized model in amaai-lab/text2midi`:**
Это ошибка при попытке загрузить через `AutoTokenizer`. В коде используется
`T5Tokenizer.from_pretrained("google/flan-t5-base")` напрямую — убедись,
что используется актуальная версия `Text2Prefix/text2prefix_midi.py`.

**`torch.load` выдаёт предупреждение о `weights_only`:**
Ожидаемо — модель загружается полностью (`weights_only=False`). Не критично.

**Файлы моделей имеют расширение `.txt`:**
`torch.save` сохраняет ZIP-архив; расширение файла не важно для `torch.load`.
Передавай путь как есть: `data/meta_info/.../model_30_epochs.txt`.

