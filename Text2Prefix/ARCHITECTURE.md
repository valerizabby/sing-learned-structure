# Text2Prefix: Text → MIDI Piano Roll

## Задача

Сгенерировать короткий музыкальный отрезок (8–16 баров) в формате piano roll
`[T, 128]`, совместимом с SingLS, по текстовому описанию.

---

## Рекомендуемый модуль: `text2prefix_midi.py` (text2midi)

TODO проверить можно ли ускорить инференс? нужно ли генерировать так много тактов?
TODO решить проблему коллапсирования нот слева? возможно следует брать меньше тактов? проработать температуру?

Прямая генерация MIDI без аудио-промежуточного шага.

```
Текстовый промт
"upbeat jazz piano, 120 BPM"
        │
        ▼
┌───────────────────────┐
│  text2midi            │  amaai-lab/text2midi (HuggingFace)
│  LLaMA-based decoder  │  text → MIDI-токены (REMI+)
│  ~400 MB              │  → pretty_midi.PrettyMIDI
└───────────────────────┘
        │  pretty_midi
        ▼
┌───────────────────┐
│  Compatibility    │  Согласование с форматом SingLS
│  Layer            │  • MIDI → piano roll [T, 128]
│                   │  • binarize (threshold=0.5)
│                   │  • clip pitch to [20, 108]
│                   │  • align frame rate: fs = tempo / 60
└───────────────────┘
        │
        ▼
(piano_roll [T, 128], tempo: float, num_beats: int)
```

### Установка

```bash
pip install transformers>=4.40 miditok>=3.0 pretty_midi huggingface_hub accelerate
```

---

## Устаревший модуль: `text2prefix.py` (MusicGen + BasicPitch)

Оставлен для совместимости. Используется как fallback, если text2midi недоступен.

```
Текстовый промт
        │
        ▼
┌───────────────────┐
│  MusicGen-small   │  Meta AI, text-to-audio → wav (mono, 32kHz)
└───────────────────┘
        │  wav
        ▼
┌───────────────────┐
│   Basic Pitch     │  Spotify, audio → MIDI (нейросетевая транскрипция)
└───────────────────┘
        │  MIDI
        ▼
┌───────────────────┐
│  Compatibility    │  Согласование с форматом SingLS
│  Layer            │
└───────────────────┘
        │
        ▼
(piano_roll [T, 128], tempo: float, num_beats: int)
```

---

## Сравнение подходов

| Критерий               | text2prefix_midi.py (✅ рекомендуется) | text2prefix.py (⚠️ устарел) |
|------------------------|----------------------------------------|------------------------------|
| Шагов в pipeline       | 2: text → MIDI → piano roll            | 3: text → audio → MIDI → roll |
| Точность нот           | ✅ точная — нет транскрипции           | ❌ BasicPitch вносит шум     |
| Контроль темпа         | ✅ точный (MIDI tempo events)          | ❌ BPM влияет приблизительно |
| Зависимости            | ✅ лёгкие (transformers, miditok)      | ❌ audiocraft (тяжёлый), xformers-конфликты |
| Домен                  | символьная музыка (MIDI)               | аудио (любые тембры)         |

---

## Ключевые компоненты `text2prefix_midi.py`

### text2midi (amaai-lab)
- Модель: `amaai-lab/text2midi` (~400 MB, CUDA/MPS/CPU)
- Архитектура: LLaMA-based decoder, обучен на парах (текст, MIDI)
- MIDI-токенизация: REMI+ (miditok)
- Вход: текстовый промт + `{N} BPM`
- Выход: последовательность MIDI-токенов → `pretty_midi.PrettyMIDI`

### Compatibility Layer (идентичен text2prefix.py)

| Параметр | SingLS | text2prefix_midi |
|----------|--------|-----------------|
| Piano roll shape | `[T, 128]` | `[T, 128]` ✓ |
| Frame rate | `fs = tempo/60` bps | вычисляется из tempo ✓ |
| Note range | 20–108 | clip → [20, 108] ✓ |
| Binarized | да | порог 0.5 ✓ |
| Формат вывода | `(tensor, tempo, num_beats)` | `(tensor, tempo, num_beats)` ✓ |

---

## Интерфейс модуля

```python
from Text2Prefix.text2prefix_midi import Text2PrefixMIDI

gen = Text2PrefixMIDI()   # загрузка модели при первом вызове .generate()

piano_roll, tempo, num_beats = gen.generate(
    prompt    = "gentle piano melody in C major",
    n_bars    = 8,
    tempo     = 120.0,     # явно передаём темп — повышает воспроизводимость
)
# piano_roll: Tensor [T, 128]
# tempo:      float (BPM)
# num_beats:  int
```

---

## Файловая структура

```
Text2Prefix/
├── ARCHITECTURE.md         ← этот файл
├── text2prefix_midi.py     ← ✅ рекомендуемый модуль (text2midi)
├── text2prefix.py          ← ⚠️  устаревший модуль (MusicGen + BasicPitch)
├── requirements.txt        ← зависимости для text2prefix_midi.py
└── demo.py                 ← быстрая проверка pipeline
```

---

## Интеграция в end-to-end пайплайн

```python
from Text2Prefix.text2prefix_midi import Text2PrefixMIDI
from Seg2SSM.affinity_ssm          import AffinitySSM

# Шаг 1: генерируем prefix из текста
prefix_roll, tempo, num_beats = Text2PrefixMIDI().generate(
    prompt = "melancholic piano, slow tempo",
    n_bars = 8,
    tempo  = 80.0,
)

# Шаг 2: строим SSM из сегментного плана
segment_plan = [(0,8), (1,16), (2,16), (1,16), (2,16), (5,8)]
ssm = AffinitySSM.fixed().build(segment_plan, ssm_size=64)

# Шаг 3: SingLS генерирует полный трек
# (передаём prefix_roll и ssm в SingLS inference)
```
