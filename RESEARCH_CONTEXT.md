# RESEARCH_CONTEXT.md — База знаний по MIR и генерации музыки

Справочный документ для работы с репозиторием. Описывает домены, репрезентации, датасеты, модели и метрики, актуальные для задачи structure-aware music generation.

---

## 1. Два домена: audio и symbolic/MIDI

**Audio-domain** модели работают с waveform, spectrogram или latent-audio токенами.
- Сильная сторона: тембр, фактура, реалистичность звучания.
- Слабая сторона: точный контроль структуры, формы, повторов.

**Symbolic/MIDI-domain** модели работают с нотными событиями: pitch, duration, velocity, bar, position, chord, instrument.
- Сильная сторона: композиционная логика, форма, повторяемость, редактируемость.
- Слабая сторона: нужен этап рендеринга в audio; тембр теряется без продуманного пайплайна.

**Вывод для этого репозитория:** symbolic-домен — основное пространство планирования структуры; audio-домен — этап рендеринга, style transfer или text-conditioning.

---

## 2. MIR как источник условий и метрик

MIR-задачи, релевантные для данного проекта:

| Задача | Зачем нужна |
|--------|-------------|
| Music structure analysis (MSA) | Границы секций, типы, повторяемость → structural prior |
| Beat/downbeat/chord/key tracking | Ритмический и гармонический каркас → conditioning |
| Automatic music transcription (AMT) | Audio → MIDI → symbolic training corpus |
| Source separation | Выделение stems для multi-track conditioning |
| Representation learning | Универсальные embeddings → evaluation и retrieval |

---

## 3. Structure analysis и SSM в audio-домене

**Классический MIR-пайплайн:**
```
feature extractor → SSM → novelty / DP / block-matching → section boundaries
```

**Learned SSM пайплайн:**
```
encoder (learns bar-level embeddings) → SSM → block geometry optimization
```

Ключевые идеи из литературы:
- CBM (Correlation Block-Matching): barwise analysis без тяжёлой supervision.
- Peeters 2023: SSM-loss и novelty-loss как часть функции потерь, а не только постобработка.

**Вывод для репозитория:** перед тем как "генерировать форму", можно сначала научить encoder производить представления, где SSM имеет нужную блочную геометрию. Это прямое продолжение линии SingLS/LSA.

---

## 4. Audio embeddings и foundation models

| Модель | Назначение |
|--------|------------|
| **MERT** | Self-supervised music encoder, first choice для audio признаков |
| **CLAP** | Мультимодальный audio-text alignment, retrieval, оценка text-audio соответствия |

Для структурного контроля foundation embeddings мало помогают без barwise/segmentwise objectives.

---

## 5. AMT: audio → MIDI

| Модель | Когда использовать |
|--------|-------------------|
| **MT3** | Research-grade unified transcription, multi-track, несколько датасетов |
| **Basic Pitch** | Быстрый практичный baseline, instrument-agnostic, удобная интеграция |

Для пайплайна "audio dataset → symbolic training corpus": Basic Pitch как bootstrapping baseline, MT3 — для полифонии и multi-instrument.

---

## 6. Source separation

**HT Demucs** — наиболее практичный open baseline. Гибридный temporal/spectral U-Net с cross-domain transformer blocks.

Полезен для:
- Выделения вокала/ритм-секции для conditioning.
- Подготовки multi-track symbolic targets.
- Более чистых feature streams для structure/chord/beat extraction.

---

## 7. Событийные репрезентации в symbolic-домене

| Тип токенизации | Описание | Когда использовать |
|-----------------|----------|-------------------|
| Flat event | NOTE_ON, NOTE_OFF, TIME_SHIFT, VELOCITY | Простые задачи, но плохо кодирует форму |
| **REMI-style** | BAR, POSITION, TEMPO, CHORD + note events | Контроль структуры, ритм, гармония |
| Compound | Несколько атрибутов в одном macro-token | Много атрибутов на событие, длинные последовательности |

Если в коде используется "сырое" MIDI flattening без явных bar/position tokens — это первый кандидат на пересмотр при задаче структурного контроля.

---

## 8. Датасеты symbolic-домена

| Датасет | Треков | Особенности |
|---------|--------|-------------|
| **MAESTRO** | ~1200 | Piano audio+MIDI aligned, экспрессивное исполнение |
| **POP909** | 909 | Pop songs + chord/beat/key/bars — лучший для structure |
| **LMD** | ~170k | Большой, но шумный; требует препроцессинга |
| **MidiCaps** | 168k+ | MIDI + текстовые описания (tempo, chords, genre, mood) |

Для задач структуры: POP909 и curated subsets LMD предпочтительнее масштабного, но грязного LMD целиком.

---

## 9. Модели генерации в audio-домене

| Модель | Тип | Практическая ценность |
|--------|-----|-----------------------|
| **MusicGen** | AR Transformer + EnCodec | Open practical baseline, melody conditioning |
| **Stable Audio Open** | Latent diffusion | Open weights, до 47 сек стерео, inference-time editing |
| **AudioLDM 2** | Latent diffusion + AudioMAE | Промежуточное аудио-представление |
| **Mustango** | LDM + MuNet guidance | Music-aware control: key, tempo, chords, beats |
| **MusicLM** | Hierarchical AR | Концептуальный референс иерархии; не open |
| **Jukebox** | Multi-scale VQ-VAE + AR | Исторический референс raw-audio generation |

**DITTO** — inference-time noise-latent optimization под differentiable feature losses; шаблон для внешнего control loop без переобучения.

---

## 10. Модели генерации в symbolic-домене

| Модель | Вклад |
|--------|-------|
| **Music Transformer** | Relative attention для long-context symbolic generation |
| **Pop Music Transformer / REMI** | Стандарт bar/position/chord токенизации |
| **Museformer** | Fine-grained + coarse-grained attention, связь нот и блоков |
| **Nested Music Transformer** | Compound tokens: main decoder + sub-decoder |
| **SING** | SSM как attention prior (база этого репозитория) |

---

## 11. Оценка качества

### Audio-domain
- **FAD (Fréchet Audio Distance)** — основная автоматическая метрика.
- **CLAP similarity** — text-audio alignment для text-conditioned generation.
- **Listening studies** — обязательны; одной автоматической метрикой проблему не закрыть.

### Symbolic-domain
Рекомендуемый минимальный набор для structure-aware generation:

| Метрика | Что измеряет |
|---------|-------------|
| Section boundary P/R/F1 | Точность обнаружения границ секций |
| Segment label consistency | Правильность типов секций (A/B/A/B) |
| SSM similarity (IOU, MSE) | Структурное сходство генерации и оригинала |
| Pitch-class / note density | Дистрибутивные symbolic метрики |
| FMD (Fréchet Music Distance) | Symbolic embeddings, аналог FAD |

Один IOU недостаточен — нужен набор метрик с разных уровней.

---

## 12. Что смотреть в коде при работе с репозиторием

### Для symbolic generation (этот репо)

- [ ] Какая токенизация: flat, REMI, compound?
- [ ] Есть ли bar/position/chord tokens?
- [ ] Есть ли multi-scale attention или explicit section tokens?
- [ ] Как кодируются повторяющиеся сегменты?
- [ ] Можно ли предсказывать SSM или section graph отдельно от note decoder?

### Перспективные точки улучшения (в порядке приоритета)

1. **Attention bias от SSM** — soft additive logit, а не hard masking.
2. **Двухэтапность** — structure plan → note realization.
3. **Bar-level memory tokens** — явные узлы для крупной формы.
4. **Joint objective** — LM loss + SSM/novelty/boundary loss.
5. **Retrieval-augmented prompting** — structural template из корпуса.

---

## 13. Наиболее жизнеспособная линия для данной дипломной темы

```
symbolic transformer
    + explicit structural prior (SSM as target/bias/auxiliary prediction)
    + MIR-based validation (boundary F1, SSM similarity, FMD)
    + audio rendering (опционально: FAD/CLAP/listening study)
```

**Против end-to-end text-to-audio как основной линии:** эффектен, но хуже отвечает на научный вопрос "генерация с заданными структурными параметрами". Symbolic-first подход концептуально чище для задачи structure control.

---

## 14. Shortlist open baselines по задачам

**Audio MIR**
- Embeddings: MERT
- Text-audio alignment: CLAP
- Beat/downbeat: Beat This! / BEAST (streaming)
- AMT: MT3 / Basic Pitch
- Source separation: HT Demucs

**Audio generation**
- Open practical baseline: MusicGen
- Diffusion/open weights: Stable Audio Open
- Music-feature control: Mustango
- Conceptual references: AudioLDM 2, MusicLM, Jukebox

**Symbolic/MIDI**
- Long-context baseline: Music Transformer
- Representation baseline: Pop Music Transformer / REMI
- Structure-aware attention: Museformer
- SSM/structure bias: SING
- Compound decoding: Nested Music Transformer

**Датасеты**
- Piano performance: MAESTRO
- Pop structure: POP909
- Scale: LMD (с осторожностью)
- Text-MIDI: MidiCaps
- Text-audio: MusicCaps
