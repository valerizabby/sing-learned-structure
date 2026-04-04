# Pipeline — инструкция по запуску

Все команды запускаются из **корня репозитория** (`sing-learned-structure/`).

---

## Доступные модели

Чекпоинты лежат в `data/meta_info/` с расширением `.txt` — это обычные
PyTorch-файлы (`torch.save`), расширение не влияет на загрузку.

| Директория | Описание | IOU ↑ | MSE ↓ |
|------------|----------|-------|-------|
| `trained_transformer_original_less_struct_combined` | **Лучшая по IOU** — HierarchicalGenerator, β=0.03 | 0.2751 | 0.1007 |
| `trained_lsa_combined` | LSA attention | 0.1450 | 0.0682 |
| `trained_original_combined` | SING original attention | 0.2006 | 0.1628 |

Путь к чекпоинту: `data/meta_info/<название>/model_30_epochs.txt`

---

## Быстрый старт

### Генерация одного трека

```bash
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody in C major, slow tempo" \
    --segment "intro:8,verse:16,chorus:16,verse:16,chorus:16,outro:8" \
    --tempo 90 \
    --n_prefix_bars 8 \
    --n_gen_bars 64 \
    --ssm_type affinity \
    --warmup_beats 10 \
    --out_dir outputs/my_run
```

**Результаты** сохранятся в `outputs/my_run/`:
- `generated.mid` — полный MIDI-трек (prefix + generated)
- `prefix.mid` — только prefix (для сравнения)
- `pipeline_overview.png` — 3 панели: prefix piano roll | full piano roll | SSM heatmap

---

### Ablation study (все 4 типа SSM за один запуск)

```bash
python -m pipeline.ablation \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody in C major, slow tempo" \
    --segment "intro:8,verse:16,chorus:16,verse:16,chorus:16,outro:8" \
    --tempo 90 \
    --n_prefix_bars 8 \
    --n_gen_bars 64 \
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
| `--prompt` | — | **Обязательный.** Текстовый промт для MusicGen |
| `--segment` | — | **Обязательный.** Сегментный план (см. ниже) |
| `--tempo` | `120.0` | Темп в BPM |
| `--n_prefix_bars` | `8` | Длина prefix в барах |
| `--n_gen_bars` | `64` | Длина генерируемой части в барах |
| `--ssm_type` | `affinity` | Тип SSM: `affinity` / `empirical` / `random` / `none` |
| `--empirical_ckpt` | `Seg2SSM/checkpoints/affinity_matrix.pt` | Матрица A для `empirical` |
| `--warmup_beats` | `10` | Сколько последних битов prefix пускать через LSTM (см. ниже) |
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
# Классическая поп-структура
--segment "intro:8,verse:16,chorus:16,verse:16,chorus:16,bridge:8,chorus:16,outro:8"

# Короткий тест
--segment "verse:8,chorus:8,verse:8"

# С инструментальным разделом
--segment "intro:8,verse:16,chorus:16,instr:16,chorus:16,outro:8"
```

**Важно:** сумма баров в сегментном плане должна равняться `n_prefix_bars + n_gen_bars`.

---

## Параметр `--warmup_beats`

Модель SingLS обучалась на данных MAESTRO/LMD. Prefix от MusicGen+BasicPitch —
OOD (out-of-distribution) данные с другим распределением нот. Если пропустить весь
prefix через LSTM, hidden state уходит в патологическую область и генерация
коллапсирует к 2–3 повторяющимся нотам.

**Решение:** через LSTM пропускаются только последние `warmup_beats` битов prefix.
Полный prefix при этом по-прежнему доступен SSM attention-механизму.

| `--warmup_beats` | Эффект |
|-----------------|--------|
| `4–8` | Минимальное OOD-воздействие, максимальная стабильность |
| `10` (по умолчанию) | Хороший баланс для большинства промтов |
| `= n_prefix_bars × 4` | Полный prefix через LSTM (часто даёт коллапс) |

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
    --segment "intro:4,verse:16,chorus:16,verse:16,chorus:16,outro:4" \
    --tempo 120 \
    --ssm_type empirical \
    --empirical_ckpt Seg2SSM/checkpoints/affinity_matrix.pt \
    --out_dir outputs/empirical_run
```

---

## Устранение проблем

**Генерация коллапсирует к 2–3 повторяющимся нотам:**
Уменьши `--warmup_beats`:
```bash
--warmup_beats 4
```
Или сократи prefix: `--n_prefix_bars 2`.

**MusicGen не загружается:**
```bash
pip install audiocraft
```

**Basic Pitch не установлен:**
```bash
pip install basic-pitch
```

**Ошибка импорта `sparsemax`:**
```bash
pip install sparsemax
```

**`torch.load` выдаёт предупреждение о `weights_only`:**
Это ожидаемо — модель загружается полностью (`weights_only=False`).

**Файлы моделей имеют расширение `.txt`, это нормально:**
`torch.save` сохраняет ZIP-архив; расширение файла не важно для `torch.load`.
Передавай путь как есть: `data/meta_info/.../model_30_epochs.txt`.

**Слишком мало нот в prefix (density < 0.01):**
Basic Pitch мог плохо транскрибировать аудио. Попробуй другой промт или явно укажи темп.

**`max_notes = 0` при генерации:**
Сумма баров в `--segment` не совпадает с `n_prefix_bars + n_gen_bars`.
Убедись, что сумма баров в плане равна этим двум значениям.
