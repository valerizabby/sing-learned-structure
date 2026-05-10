# STATUS.md — Текущее состояние репозитория

**Дата:** 2026-05-03  
**Дипломная работа:** "Генерация музыки с заданными структурными параметрами"

---

## 1. Что сделано: общий итог

Реализован **end-to-end пайплайн генерации MIDI** с двумя каналами управления:
- **Структурный план** (`"intro:8,verse:16,chorus:16,..."`) → SSM через AffinitySSM
- **Текстовый промт** (`"gentle piano melody"`) → музыкальный prefix через text2midi

Пайплайн работает одной командой, поддерживает ablation study, сохраняет MIDI и визуализации.

---

## 2. Архитектура системы

```
Текстовый промт           Сегментный план
       ↓                  "verse:16,chorus:16,..."
 [Text2Prefix]                    ↓
  text2midi (LLaMA)         [Seg2SSM]
       ↓                    AffinitySSM
  prefix_roll                     ↓
  [T_prefix, 128]           SSM [T_total, T_total]
          ↓_______________________|
               [SingLS / 3SING*]
          HierarchicalGenerator
          LSTM + StructureTransformer
                    ↓
             full_roll [T_total, 128]
                    ↓
              generated.mid  (только T_gen часть)
              prefix.mid     (отдельно)
              pipeline_overview.png
```

### Модули

| Модуль | Путь | Статус |
|--------|------|--------|
| Основная модель | `SingLS/` | ✅ обучена |
| Seg2SSM | `Seg2SSM/` | ✅ реализован, валидирован |
| Text2Prefix | `Text2Prefix/` | ✅ работает (text2midi backend) |
| Pipeline | `pipeline/` | ✅ работает |
| Ablation | `pipeline/ablation.py` | ✅ работает |
| Eval SSM-following | `experiments/eval_ssm_following.py` | ✅ прогнан |
| UI | `ui/` | ✅ запускается через `run_ui.sh` |

---

## 3. Модели и метрики

### Финальные метрики (combined датасет, 2600 треков, temperature=1.5)

> Пересчитаны 2026-04-04 после фикса температуры. Старые числа (до фикса) недействительны.

| Модель | IOU ↑ | MSE ↓ | ScaleCons ↑ | Чекпоинт |
|--------|-------|-------|-------------|----------|
| None (baseline) | 0.0848 | 0.0645 | — | — |
| SING (original) | 0.2986 | 0.0800 | 0.5342 | `data/meta_info/trained_original_combined/` |
| LSA | 0.0959 | 0.0568 | 0.4561 | `data/meta_info/trained_lsa_combined/` |
| **3SING*** | **0.3198** | **0.0692** | **0.5330** | `data/meta_info/trained_transformer_original_less_struct_combined/` |

**Лучшая модель: 3SING*** — `HierarchicalGenerator` с `beta=0.03`, структурный лосс как регуляризация.

### Что такое 3SING*

```python
HierarchicalGenerator(
    generator = MusicGenerator(
        attention_type = AttentionType.ORIGINAL   # sparsemax по SSM
    ),
    structure_model = StructureModel(
        transformer = StructureTransformer(d_model=128, nhead=4, num_layers=2),
        proj = Linear(128, 128)
    ),
    alpha = 0.05   # вклад структурного трансформера
)
# Итоговый лосс: BCE + SSM_L2 + 0.03 * structure_loss
```

LSTM отвечает за локальную связность нот, StructureTransformer — за глобальный структурный план. Ключевой вывод из экспериментов: маленький `beta=0.03` и неизменяемый `alpha=0.05` работают лучше обучаемых версий.

---

## 4. Seg2SSM — построение SSM из сегментного плана

### Как работает

```
segment_plan → bar_labels [T] → A[bar_labels][:, bar_labels] → [T,T]
             → gaussian_blur(σ=1.5) → noise(std=0.04) → rescale → resize → [ssm_size, ssm_size]
```

### Affinity Matrix A (A_fixed, из теории музыки)

|        | intro | verse | chorus | bridge | instr | outro |
|--------|-------|-------|--------|--------|-------|-------|
| intro  | 1.00  | 0.60  | 0.40   | 0.20   | 0.50  | 0.60  |
| verse  | 0.60  | 1.00  | 0.70   | 0.25   | 0.50  | 0.35  |
| chorus | 0.40  | 0.70  | 1.00   | 0.30   | 0.55  | 0.45  |
| bridge | 0.20  | 0.25  | 0.30   | 1.00   | 0.25  | 0.20  |
| outro  | 0.60  | 0.35  | 0.45   | 0.20   | 0.40  | 1.00  |

Второй режим — **A_empirical**: оценена из 422 треков SALAMI как среднее chroma-SSM по парам меток. Хранится в `Seg2SSM/checkpoints/affinity_matrix.pt`.

### Зачем постобработка

Модель обучена на chroma-SSM с `mean≈0.50, std≈0.13`. Сырая блочная матрица имеет `mean≈0.70, std≈0.22` — другое распределение, sparsemax ведёт себя некорректно. Rescale приводит к обучающему распределению без разрушения блочной структуры.

---

## 5. SSM-following: подтверждение работоспособности

**Эксперимент** (2026-04-29, N=100 планов, n_gen_bars=64):

| Условие | IOU ↑ | MSE ↓ |
|---------|-------|-------|
| affinity | 0.0329 ± 0.0079 | 0.2877 ± 0.0415 |
| none | 0.0280 ± 0.0081 | 0.2934 ± 0.0413 |

**ΔIOU = +0.0049 ✓, ΔMSE = −0.0057 ✓** — AffinitySSM статистически улучшает структурное следование.

**Важно:** перед получением корректных результатов были исправлены два бага:

1. **SSM alignment bug** (`pipeline/generate.py`): sequence инициализировалась как
   `prefix_t[-1:]` (длина 1) → `original_attention` читал SSM-ряды 1,2,3... (intro-регион)
   вместо T_prefix, T_prefix+1... (verse/chorus-регион).
   **Фикс:** `sequence = torch.zeros_like(prefix_t)` — нулевые фреймы длиной T_prefix.

2. **StructureModel 1×1 bug** (`SingLS/model/hierarchical_generator.py`):
   `build_ssm_batch(T=1, ...)` → всегда извлекал `SSM[0:1, 0:1] = 1.0` (константа).
   **Фикс:** `T = max(lstm_out.shape[0], prev_sequence.shape[0])`.

---

## 6. Ablation study

Четыре режима SSM для сравнения (запуск: `python -m pipeline.ablation`):

| SSM type | Описание | Роль |
|----------|----------|------|
| `none` | нулевая SSM | baseline |
| `random` | случайная симметричная | нижняя граница |
| `affinity` | AffinitySSM из плана | **целевой режим** |
| `empirical` | A из SALAMI | верхняя граница |

Результаты (2026-04-04): `none(0.0254) < random(0.0284) < affinity(0.0293) < empirical(0.0318)` по IOU — ожидаемый порядок подтверждён.

---

## 7. Параметры генерации (лучшая конфигурация)

Из `experiments/SAMPLING_EXPERIMENTS.md`:

```bash
--model  data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt
--temperature 1.0 --temp_start 1.5 --temp_warmup_steps 10
--topk_k 10 --n_samples 1 --note_extension 2
--warmup_beats 4 --prefix_in_sequence false
--ssm_type affinity
```

Ключевые выводы по сэмплированию:
- `note_extension=2` — главный параметр звучания: убирает стаккато, делает ноты 3 бита
- `topk_k=5` → коллапс, `topk_k=10` — минимум без коллапса
- `temperature=1.5` обязательна при обучении/оценке; при генерации демо `1.0` + scheduling
- `warmup_beats=4` — оптимум: при 0 нет контекста, при 10 OOD-заражение hidden state

---

## 8. Пройденный путь экспериментов

### Что не сработало и почему

| Эксперимент | Гипотеза | Результат | Вывод |
|-------------|----------|-----------|-------|
| Lambda search (λ=1..20 для SSM-loss) | Большой λ усилит структуру | IOU ухудшился, MSE взлетел | `ssm_err` суммировался, а не усреднялся → эффективный коэффициент = λ×batch_size, модель перестала учить ноты |
| Sigmoid перед SSM в loss | Привести к домену [0,1] | Ещё хуже | Домены были совместимы и без sigmoid; sigmoid давал большой начальный градиент |
| Обучаемый alpha в HierarchicalGenerator | Модель сама найдёт баланс | alpha → 0.0025, структура игнорировалась | Нужен фиксированный alpha=0.05 |
| LSA как замена ORIGINAL attention | QKV+SSM даст лучше IOU | IOU 0.096 vs 0.299 — провал | LSA хорош по MSE но плох по IOU; ORIGINAL attention лучше воспроизводит крупные паттерны |
| Замена LSTM на полный трансформер | Больше capacity | Нецелесообразно | SSM уже покрывает долгосрочные зависимости; 2600 треков мало для трансформера с нуля |

### Что сработало

- `HierarchicalGenerator` с `ORIGINAL` attention + `StructureTransformer` как мягкая регуляризация (`beta=0.03`) → лучший IOU=0.3198
- Фиксированный `alpha=0.05` → стабильное обучение
- `temperature=1.5` в топ-k сэмплировании → устранение коллапса генерации
- `note_extension` для легато-звучания
- `warmup_beats + prefix_in_sequence=False` → устранение OOD-деградации от text2midi

---

## 9. Что остаётся сделать (к защите)

Из `experiments/DEMO_IMPROVEMENT_PLAN.md`:

| Шаг | Суть | Статус |
|-----|------|--------|
| ~~Шаг 1~~ | Метрика SSM-following (affinity vs none) | ✅ ΔIOU=+0.0049 |
| **Шаг 2** | In-distribution prefix из датасета (SSM-matching) | ⏳ спроектирован, не реализован |
| ~~Шаг 3~~ | MIDI post-processing (sustain, velocity) | ⏳ запланирован |
| ~~Шаг 4~~ | Cherry-pick лучшего результата | ⏳ запланирован |
| ~~Шаг 5~~ | Trim prefix из итогового MIDI | ✅ уже реализован |

**Главный незакрытый риск:** OOD-prefix от text2midi всё ещё является основным источником
деградации качества демо. Шаг 2 (SSM-matching prefix из датасета) — наибольший
потенциальный выигрыш без переобучения модели.

**Критерии готовности к защите:**
- [x] Таблица SSM-following (affinity > none) — есть
- [ ] MIDI-файл, звучащий как "пьеса"
- [ ] Структурная разметка в SSM визуально соответствует плану
- [x] Демо запускается одной командой

---

## 10. Структура репозитория (активный код)

```
SingLS/
├── model/
│   ├── model.py                  # MusicGenerator — LSTM + attention selector
│   ├── original_attention.py     # sparsemax по SSM-ряду
│   ├── lsa.py                    # QKV + SSM как мультипликативная маска
│   ├── lsaSB.py                  # QKV + SSM как аддитивный логит-бонус
│   ├── structure_transformer.py  # StructureTransformer + StructureModel
│   ├── hierarchical_generator.py # HierarchicalGenerator (лучшая архитектура)
│   └── utils.py                  # build_ssm_batch
├── trainer/
│   ├── train.py                  # ModelTrainer: обучение, custom_loss, generate
│   └── data_utils.py             # батчинг, SSM-вычисление, top-k сэмплирование
├── config/config.py              # DEVICE, hidden_size, AttentionType
└── main.py                       # точка входа обучения

Seg2SSM/
├── affinity_ssm.py               # AffinitySSM: segment_plan → SSM [64×64]
├── checkpoints/affinity_matrix.pt # A_empirical из 422 треков SALAMI
└── data_prep/                    # парсинг SALAMI, оценка A_empirical

Text2Prefix/
├── text2prefix_midi.py           # ✅ основной: text2midi (LLaMA-based)
└── text2prefix.py                # ⚠️ fallback: MusicGen + BasicPitch

pipeline/
├── generate.py                   # run_pipeline(): основной пайплайн
├── ablation.py                   # run_ablation(): 4 режима SSM
├── HOWTO.md                      # документация по запуску
└── COLLAPSE_DEBUG.md             # диагностика и фиксы коллапса генерации

experiments/
├── eval_ssm_following.py         # оценка SSM-following rate
├── DEMO_IMPROVEMENT_PLAN.md      # план улучшений к защите
├── SAMPLING_EXPERIMENTS.md       # подбор параметров генерации
└── LAMBDA_SEARCH_POSTMORTEM.md   # postmortem неудачного lambda search

inference/
├── main.py                       # сравнение моделей по IOU/MSE
├── compare_models.py             # iou_ssm(), mse_ssm(), piano_roll_to_midi()
└── artifacts.py                  # визуализации SSM, piano roll, MIDI
```

---

## 11. Ключевые технические решения и обоснования

| Решение | Альтернатива | Почему выбрано |
|---------|-------------|----------------|
| LSTM + SSM-attention | Полный трансформер | SSM покрывает долгосрочные зависимости; 2600 треков мало для трансформера; O(1) на шаг при инференсе |
| AffinitySSM аналитически | Нейросеть Seg2SSM | Сегментный план несёт мало информации (7 меток); интерпретируемо; можно объяснить на одном слайде |
| Piano roll [T, 128] | REMI-токенизация | Совместимость с оригинальным SING; простота; темпо-адаптивное разрешение |
| text2midi | MusicGen + BasicPitch | Нет потерь при транскрипции; точный темп; лёгкие зависимости |
| `alpha=0.05` фиксированный | Обучаемый alpha | Обучаемый alpha → 0 (структура игнорируется); фиксированный — стабильно |
| `beta=0.03` структурный loss | Большой beta или без него | Слишком большой beta разрушает BCE-обучение нот; 0.03 — регуляризация |

