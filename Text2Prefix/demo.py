"""
demo.py — быстрая проверка Text2Prefix pipeline.

Запуск:
    cd Text2Prefix
    python demo.py

Что проверяет:
    1. Загрузку MusicGen и Basic Pitch
    2. Генерацию piano roll из текстового промта
    3. Совместимость с форматом SingLS: shape, dtype, диапазон нот
    4. Качественный анализ: плотность, полифония, диапазон, тональный центр
    5. Экспорт в MIDI для прослушивания
    6. Визуализацию (сохраняет demo_piano_roll.png)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import pretty_midi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from text2prefix import Text2Prefix, NOTE_MIN, NOTE_MAX, BEATS_PER_BAR


# ---------------------------------------------------------------------------
# Конфигурация демо
# ---------------------------------------------------------------------------

# Несколько промтов генерируются за один запуск — модель загружается один раз.
EXPERIMENTS = [
    {"prompt": "gentle piano melody in C major, slow tempo", "tempo": 80.0,  "n_bars": 8},
    {"prompt": "upbeat jazz piano, 120 bpm",                 "tempo": 120.0, "n_bars": 8},
    {"prompt": "dark minor piano, dramatic",                 "tempo": 90.0,  "n_bars": 8},
]

OUT_DIR = Path(__file__).parent / "demo_outputs"


# ---------------------------------------------------------------------------
# Качественный анализ
# ---------------------------------------------------------------------------

def analyze_quality(pr: torch.Tensor, tempo: float, num_beats: int):
    """Выводит качественные метрики piano roll."""
    pr_np = pr.numpy()  # [T, 128]
    T = pr_np.shape[0]

    # Плотность: доля активных ячеек
    density = pr_np.mean()

    # Полифония: среднее количество одновременно звучащих нот
    polyphony_per_beat = pr_np.sum(axis=1)  # [T]
    mean_poly = polyphony_per_beat[polyphony_per_beat > 0].mean() if (polyphony_per_beat > 0).any() else 0
    max_poly  = polyphony_per_beat.max()

    # Фактический диапазон нот
    active_notes = np.where(pr_np.sum(axis=0) > 0)[0]
    if len(active_notes) > 0:
        note_lo, note_hi = active_notes.min(), active_notes.max()
        note_span = note_hi - note_lo
    else:
        note_lo, note_hi, note_span = 0, 0, 0

    # Тональный центр: какие pitch-классы встречаются чаще всего
    pitch_class_counts = np.zeros(12)
    for midi_note in range(128):
        pitch_class_counts[midi_note % 12] += pr_np[:, midi_note].sum()
    NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    top3_pc = np.argsort(pitch_class_counts)[::-1][:3]
    top3_str = ", ".join(f"{NOTE_NAMES[pc]}({pitch_class_counts[pc]:.0f})" for pc in top3_pc)

    # Заполненность по времени: доля битов с хотя бы одной нотой
    temporal_fill = (polyphony_per_beat > 0).mean()

    print(f"\n  --- Quality Analysis ---")
    print(f"  density        : {density:.4f}  (типично 0.03–0.15 для пианино)")
    print(f"  temporal fill  : {temporal_fill:.2%}  (доля битов с нотами)")
    print(f"  mean polyphony : {mean_poly:.1f} нот/бит  (типично 1–4)")
    print(f"  max polyphony  : {int(max_poly)} нот/бит")
    print(f"  note range     : MIDI {note_lo}–{note_hi}  (span={note_span})")
    print(f"  top pitch cls  : {top3_str}")

    # Предупреждения
    if density < 0.01:
        print("  ⚠ WARN: очень мало нот — Basic Pitch плохо транскрибировал аудио")
    if density > 0.3:
        print("  ⚠ WARN: слишком много нот — возможен шум транскрипции")
    if temporal_fill < 0.3:
        print("  ⚠ WARN: много пустых битов — проверь tempo/fs согласование")
    if max_poly > 10:
        print("  ⚠ WARN: высокая полифония — возможны артефакты Basic Pitch")

    return {
        "density": density,
        "temporal_fill": temporal_fill,
        "mean_polyphony": mean_poly,
        "note_lo": note_lo,
        "note_hi": note_hi,
    }


# ---------------------------------------------------------------------------
# Экспорт в MIDI
# ---------------------------------------------------------------------------

def save_midi(pr: torch.Tensor, tempo: float, path: Path):
    """Конвертирует piano roll в MIDI и сохраняет."""
    fs = tempo / 60.0  # фреймов/сек = долей/сек
    pr_np = pr.numpy().T  # [128, T]
    notes_count, T = pr_np.shape

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    for note in range(notes_count):
        on = None
        for t in range(T):
            if pr_np[note, t] > 0 and on is None:
                on = t
            elif pr_np[note, t] == 0 and on is not None:
                instrument.notes.append(pretty_midi.Note(
                    velocity=90,
                    pitch=note,
                    start=on / fs,
                    end=t / fs,
                ))
                on = None
        if on is not None:
            instrument.notes.append(pretty_midi.Note(
                velocity=90, pitch=note,
                start=on / fs, end=T / fs,
            ))

    midi.instruments.append(instrument)
    midi.write(str(path))
    print(f"  Saved MIDI: {path}")


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def plot_piano_roll(pr: torch.Tensor, tempo: float, num_beats: int, path: Path):
    """Сохраняет piano roll как PNG."""
    fig, ax = plt.subplots(figsize=(14, 4))
    pr_np = pr.numpy().T  # [128, T]
    ax.imshow(pr_np, aspect="auto", origin="lower", cmap="Blues",
              extent=[0, num_beats, 0, 128])
    ax.axhline(NOTE_MIN, color="orange", lw=0.8, linestyle="--", label=f"NOTE_MIN={NOTE_MIN}")
    ax.axhline(NOTE_MAX, color="red",    lw=0.8, linestyle="--", label=f"NOTE_MAX={NOTE_MAX}")
    ax.set_xlabel("Beat")
    ax.set_ylabel("MIDI note")
    ax.set_title(f'Piano Roll — tempo={tempo:.1f} BPM, T={num_beats}')
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved PNG : {path}")


# ---------------------------------------------------------------------------
# Основной тест
# ---------------------------------------------------------------------------

def run_demo():
    print("=" * 60)
    print("Text2Prefix — demo")
    print("=" * 60)

    OUT_DIR.mkdir(exist_ok=True)

    # Модель загружается один раз — все эксперименты используют один объект gen
    print("\n[init] Loading Text2Prefix (MusicGen loads on first generate)...")
    gen = Text2Prefix(model_size="small")
    print(f"  Device: {gen.device}")

    results = []
    for i, exp in enumerate(EXPERIMENTS):
        prompt  = exp["prompt"]
        n_bars  = exp["n_bars"]
        tempo   = exp["tempo"]
        tag     = f"exp{i+1}"

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(EXPERIMENTS)}] prompt : \"{prompt}\"")
        print(f"             tempo  : {tempo} BPM,  n_bars={n_bars}")

        piano_roll, tempo_out, num_beats = gen.generate(
            prompt=prompt, n_bars=n_bars, tempo=tempo,
        )

        # Совместимость с SingLS
        expected_T = n_bars * BEATS_PER_BAR
        assert piano_roll.shape == (expected_T, 128)
        assert piano_roll[:, :NOTE_MIN].sum() == 0
        assert piano_roll[:, NOTE_MAX + 1:].sum() == 0
        print(f"  shape {tuple(piano_roll.shape)}  ✓")

        # Качество
        metrics = analyze_quality(piano_roll, tempo_out, num_beats)

        # Сохранение
        plot_piano_roll(piano_roll, tempo_out, num_beats,
                        OUT_DIR / f"{tag}_piano_roll.png")
        save_midi(piano_roll, tempo_out, OUT_DIR / f"{tag}_output.mid")

        results.append((piano_roll, tempo_out, num_beats))

    print("\n" + "=" * 60)
    print(f"Done. {len(EXPERIMENTS)} experiments saved to {OUT_DIR}/")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_demo()
