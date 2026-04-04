# Диагностика коллапса генерации

Все команды запускаются из корня репозитория (`sing-learned-structure/`).

---

## Три причины коллапса

1. **OOD-prefix заражает attention** — `weight_vec` в `original_attention` — это прямая взвешенная сумма piano roll фреймов из `prev_sequence`. OOD-фреймы от BasicPitch тянут логиты в патологическую область.
2. **Петля обратной связи** — как только первые шаги дали 2-3 ноты, `prev_sequence` заполняется ими, attention указывает на те же ноты, коллапс закрепляется.
3. **AffinitySSM ≠ chroma-SSM** — обучающие SSM органичные (chroma-cosine), AffinitySSM блочно-постоянная → sparsemax даёт равномерное внимание внутри секции → размытые логиты.

---

## Шаг 1: изолировать причину — выключить prefix полностью

Цель: проверить, коллапсирует ли генерация вообще без OOD-prefix (warmup_beats=0, ssm_type=none).

```bash
python3 -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody" \
    --segment "verse:4,chorus:4,verse:4,chorus:4" \
    --n_prefix_bars 4 \
    --n_gen_bars 12 \
    --warmup_beats 0 \
    --ssm_type none \
    --out_dir outputs/debug_no_prefix_no_ssm
```

**Интерпретация:**
- Коллапс есть → проблема в самой авторегрессии (нужна температура в сэмплировании)
- Коллапса нет → проблема точно в OOD-prefix или SSM → идём к шагу 2
- ![pipeline_overview.png](../outputs/debug_no_prefix_no_ssm/pipeline_overview.png)

**Результат (2026-04-01):** коллапс подтверждён — генерация залипала на 2 нотах.
Применён Fix 1: температура 1.5 в `topk_sample_one` (`SingLS/trainer/data_utils.py`).
После фикса генерация стала разнообразной. Концентрация на 2 нотах в регистре prefix — это когерентность, не коллапс (проверено сменой prefix).

---

## Шаг 2: изолировать влияние AffinitySSM

Сравнить генерацию с `none` vs `affinity` при одинаковом warmup_beats=0.

```bash
# Вариант A: без SSM
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody" \
    --segment "verse:4,chorus:4,verse:4,chorus:4" \
    --n_prefix_bars 4 \
    --n_gen_bars 12 \
    --warmup_beats 0 \
    --ssm_type none \
    --out_dir outputs/debug_ssm_none
```

```bash
# Вариант B: с AffinitySSM
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody" \
    --segment "verse:4,chorus:4,verse:4,chorus:4" \
    --n_prefix_bars 4 \
    --n_gen_bars 12 \
    --warmup_beats 0 \
    --ssm_type affinity \
    --out_dir outputs/debug_ssm_affinity
```

**Интерпретация:**
- B хуже A → AffinitySSM + sparsemax несовместимы (блочная структура → равномерные веса → размытые логиты)
- A = B → SSM не при чём, проблема только в prefix/LSTM

**Результат (2026-04-01):** A ≈ B — генерация с AffinitySSM и без визуально сопоставима.
AffinitySSM не ломает генерацию. Блоки на матрице слабовыражены (verse↔chorus=0.7, контраст 0.3) — это ожидаемо для данного сегментного плана.

---

## Шаг 3: изолировать влияние OOD-prefix

Сравнить warmup_beats=0 vs warmup_beats=10 при ssm_type=affinity.

```bash
# Вариант A: prefix не проходит через LSTM
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody in C major, slow tempo" \
    --segment "intro:4,verse:8,chorus:8,verse:8,chorus:8,outro:4" \
    --n_prefix_bars 8 \
    --n_gen_bars 32 \
    --warmup_beats 0 \
    --ssm_type affinity \
    --out_dir outputs/debug_warmup_0
```

```bash
# Вариант B: 4 бита prefix через LSTM
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody in C major, slow tempo" \
    --segment "intro:4,verse:8,chorus:8,verse:8,chorus:8,outro:4" \
    --n_prefix_bars 8 \
    --n_gen_bars 32 \
    --warmup_beats 4 \
    --ssm_type affinity \
    --out_dir outputs/debug_warmup_4
```

```bash
# Вариант C: 10 битов (текущий дефолт)
python -m pipeline.generate \
    --model data/meta_info/trained_transformer_original_less_struct_combined/model_30_epochs.txt \
    --prompt "gentle piano melody in C major, slow tempo" \
    --segment "intro:4,verse:8,chorus:8,verse:8,chorus:8,outro:4" \
    --n_prefix_bars 8 \
    --n_gen_bars 32 \
    --warmup_beats 10 \
    --ssm_type affinity \
    --out_dir outputs/debug_warmup_10
```

**Интерпретация:**
- A лучше B лучше C → warmup_beats надо уменьшать, в идеале = 0
- A = B = C → дело не в warmup, а в самом prefix как части `prev_sequence` (attention его видит всегда)

**Результат (2026-04-01):**
- warmup=0 — широкий диапазон нот, но слишком хаотично (нет музыкального контекста)
- warmup=4 — лучший компромисс: prefix прогревает LSTM, но не вызывает коллапс
- warmup=10 — коллапс: OOD-фреймы искажают hidden state, генерация залипает на 2-3 нотах
Рекомендуемый дефолт: `warmup_beats=4`.

---

## Шаг 4: диагностика качества prefix

Добавь в `pipeline/generate.py` после строки `print(f"  prefix shape ...")` (строка ~317):

```python
density = prefix_roll.mean().item()
active_notes = int((prefix_roll.sum(dim=0) > 0).sum().item())
print(f"  Prefix density  : {density:.4f}  (ожидается 0.05–0.15)")
print(f"  Active notes    : {active_notes}/128  (ожидается 30–60)")
```

**Интерпретация:**
- density > 0.3 или active_notes > 80 → BasicPitch даёт слишком плотный piano roll (OOD)
- density < 0.01 → BasicPitch почти ничего не транскрибировал → prefix почти пустой

---

## Потенциальные фиксы (после диагностики)

### Fix 1: Температура в сэмплировании
Если коллапс воспроизводится даже при warmup_beats=0 — добавить температуру.

В `SingLS/trainer/data_utils.py`, функция `topk_sample_one` (строка ~134):
```python
# Было:
seq = torch.distributions.Categorical(softmax(vals.float()))

# Стало (temperature=1.5 делает распределение равномернее):
seq = torch.distributions.Categorical(softmax((vals / 1.5).float()))
```

### Fix 2: Prefix не участвует в attention
В `pipeline/generate.py`, функция `generate_from_prefix`:
- Инициализировать `sequence` как пустой тензор `[0, 1, 128]`
- Prefix пускать только через LSTM (warmup), но **не** добавлять в `sequence`
- Тогда attention видит только сгенерированные in-distribution ноты

### Fix 3: Warm start из обучающих данных
Вместо Text2Prefix → взять случайный кусок из `data/combined/combined_test.pt` как prefix.
Это гарантирует in-distribution hidden state и `prev_sequence`.
Text2Prefix при этом используется только как описание — структурный план и SSM.

---

## Ожидаемый порядок проверки

1. Шаг 1 → если нет коллапса: проблема в prefix/SSM → Шаг 2 + 3
2. Шаг 1 → если коллапс есть: сразу Fix 1 (температура), потом повтор
3. Шаг 4 → всегда полезен, добавить до начала тестов
