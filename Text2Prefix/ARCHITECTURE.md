# Text2Prefix: Text → MIDI Piano Roll

## Задача

Сгенерировать короткий музыкальный отрезок (8–16 баров) в формате piano roll
`[T, 128]`, совместимом с SingLS, по текстовому описанию.

---

## Pipeline

```
Текстовый промт
"upbeat jazz piano, 120 bpm"
        │
        ▼
┌───────────────────┐
│  MusicGen-small   │  Meta AI, text-to-audio
│  facebook/        │  → wav (mono, 32kHz)
│  musicgen-small   │  длина: n_bars × 4 / (tempo/60) сек
└───────────────────┘
        │  wav (numpy array)
        ▼
┌───────────────────┐
│   Basic Pitch     │  Spotify, audio → MIDI
│   (нейросеть)     │  → список нот (onset, offset, pitch, velocity)
└───────────────────┘
        │  note events
        ▼
┌───────────────────┐
│  Compatibility    │  Согласование с форматом SingLS
│  Layer            │  • tempo detection (librosa)
│                   │  • MIDI → piano roll [T, 128]
│                   │  • binarize (threshold=0.5)
│                   │  • clip pitch to [20, 108]
│                   │  • align frame rate: fs = tempo / 60
└───────────────────┘
        │
        ▼
(piano_roll [T, 128], tempo: float, num_beats: int)
```

---

## Ключевые компоненты

### MusicGen (Meta AI)
- Модель: `facebook/musicgen-small` (~300 MB, работает на CPU/MPS/CUDA)
- Вход: текстовый промт + длительность в секундах
- Выход: аудио tensor (stereo → усредняем до mono)
- Зависимость: `audiocraft` (pip)

### Basic Pitch (Spotify)
- Нейросетевая транскрипция audio → MIDI
- Вход: numpy array (mono, любая частота дискретизации)
- Выход: `pretty_midi.PrettyMIDI` с нотными событиями
- Зависимость: `basic-pitch` (pip)

### Compatibility Layer
Согласование с форматом SingLS:

| Параметр | SingLS | Text2Prefix |
|----------|--------|-------------|
| Piano roll shape | `[T, 128]` | `[T, 128]` ✓ |
| Frame rate | `fs = tempo/60` bps | вычисляется из tempo |
| Note range | 20–108 | clip → [20, 108] ✓ |
| Binarized | да | порог 0.5 ✓ |
| Формат вывода | `(tensor, tempo, num_beats)` | `(tensor, tempo, num_beats)` ✓ |

**Frame rate:** SingLS использует `fs = tempo/60` фреймов в секунду
(один фрейм = одна доля). При tempo=120 → fs=2 фрейма/сек → T = n_bars × 4 × 2 = 64 для 8 баров.

---

## Интерфейс модуля

```python
from text2prefix import Text2Prefix

gen = Text2Prefix(model_size="small")   # или "medium"

piano_roll, tempo, num_beats = gen.generate(
    prompt    = "gentle piano melody in C major",
    n_bars    = 8,
    tempo     = 120.0,     # None = авто-определение из аудио
)
# piano_roll: Tensor [T, 128]
# tempo:      float (BPM)
# num_beats:  int   (T × beats_per_frame = num_beats)
```

---

## Файловая структура

```
Text2Prefix/
├── ARCHITECTURE.md      ← этот файл
├── text2prefix.py       ← основной модуль (Text2Prefix)
├── requirements.txt     ← зависимости
└── demo.py              ← быстрая проверка pipeline
```

---

## Ограничения и компромиссы

| Ограничение | Следствие |
|---|---|
| MusicGen генерирует аудио (не MIDI) | транскрипция Basic Pitch вносит шум |
| Basic Pitch оптимизирован под piano | для других инструментов качество ниже |
| MusicGen не управляет точным темпом | tempo в промте влияет приблизительно |
| Tempo авто-детект может ошибаться | рекомендуется явно передавать tempo |

**Рекомендация для диплома:** указывать tempo явно в промте и передавать параметром,
это даёт воспроизводимые результаты.

---

## Интеграция в end-to-end пайплайн

```python
from Text2Prefix.text2prefix import Text2Prefix
from Seg2SSM.affinity_ssm    import AffinitySSM

# Шаг 1: генерируем prefix из текста
prefix_roll, tempo, num_beats = Text2Prefix().generate(
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
