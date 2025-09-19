
import pandas as pd
import pretty_midi
import numpy as np
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    # === Настройки ===
    INPUT_PARQUET = "tracks_with_aligned_midi.parquet"
    OUTPUT_PICKLE = "lmd_processed.pkl"

    MIN_BEATS = 4  # Минимум тактов, иначе отбрасываем трек

    # === Загрузка данных ===
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} tracks")

    # === Предобработка ===
    processed = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing MIDI"):
        path = row["midi_filename"]
        try:
            midi_data = pretty_midi.PrettyMIDI(path)
        except Exception as e:
            continue  # битый MIDI — пропускаем

        try:
            tempo = midi_data.estimate_tempo()
            roll = midi_data.get_piano_roll(fs=tempo / 60).T  # [beats, 128]
            if roll.shape[0] < MIN_BEATS:
                continue

            # Удаление тишины спереди
            while roll.shape[0] > 0 and np.all(roll[0] == 0):
                roll = roll[1:]

            # Удаление тишины сзади
            while roll.shape[0] > 0 and np.all(roll[-1] == 0):
                roll = roll[:-1]

            # Бинаризация
            roll = np.where(roll > 0, 1, 0)
            beats = roll.shape[0]

            processed.append({
                "track_id": row.get("track_id"),
                "title": row.get("canonical_title"),
                "composer": row.get("canonical_composer"),
                "year": row.get("year"),
                "duration": row.get("duration"),
                "tempo": tempo,
                "beats": beats,
                "split": row.get("split"),
                "roll": roll
            })

        except Exception as e:
            continue  # на случай непредвиденных ошибок

    # === Сохранение ===
    processed_df = pd.DataFrame(processed)
    processed_df.to_pickle(OUTPUT_PICKLE)
    print(f"Saved {len(processed_df)} processed tracks to {OUTPUT_PICKLE}")
