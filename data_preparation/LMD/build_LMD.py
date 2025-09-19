from pathlib import Path
from typing import Optional

import pandas as pd
import pretty_midi
from tqdm import tqdm

# Пути и настройки
LMD_ROOT = Path("/Users/valerizab/Desktop/LMD-dataset/lmd_aligned")
LMD_AUDIO_ROOT = Path("/Users/valerizab/Desktop/LMD-dataset/lmd_matched_mp3")
NUM_TRACKS = 1300
PARQUET_INPUT = "train-00000-of-00001.parquet"
PARQUET_OUTPUT = "tracks_with_aligned_midi.parquet"

# Переименование колонок для приведения к MAESTRO-стилю
MAPPING = {
    "title": "canonical_title",
    "artist": "canonical_composer",
}


def get_midi_duration(path: str) -> Optional[float]:
    try:
        midi = pretty_midi.PrettyMIDI(path)
        return midi.get_end_time()
    except Exception:
        return None


def find_any_aligned_midi(track_id: str) -> Optional[str]:
    subdir = LMD_ROOT / track_id[2] / track_id[3] / track_id[4] / track_id
    if not subdir.exists():
        return None
    midi_files = sorted(subdir.glob("*.mid"))
    return str(midi_files[0]) if midi_files else None


def find_any_matched_audio(track_id: str) -> Optional[str]:
    subdir = LMD_AUDIO_ROOT / track_id[2] / track_id[3] / track_id[4]
    if not subdir.exists():
        return None
    candidates = sorted(subdir.glob(f"{track_id}*.mp3"))
    return str(candidates[0]) if candidates else None


def align_dataset_parser(df: pd.DataFrame, path_to_save: str) -> None:
    tqdm.pandas(desc="Поиск путей и длительности")

    # Генерация путей
    df["midi_filename"] = df["track_id"].progress_apply(find_any_aligned_midi)
    df["audio_filename"] = df["track_id"].progress_apply(find_any_matched_audio)

    # Удаляем строки без MIDI
    df = df[df["midi_filename"].notna()]
    print(f"Found {len(df)} matching aligned MIDI files")

    # Длительность
    df["duration"] = df["midi_filename"].progress_apply(get_midi_duration)

    # Разделение на train/test
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.9)
    df.loc[:split_idx, "split"] = "train"
    df.loc[split_idx:, "split"] = "test"

    # Переименование колонок
    df.rename(columns=MAPPING, inplace=True)

    # Сохраняем
    df.to_parquet(path_to_save, index=False)
    print(f"Saved to {path_to_save}")


if __name__ == "__main__":
    # Загрузка исходного parquet
    df = pd.read_parquet(PARQUET_INPUT)
    print(f"Loaded {len(df)} track metadata entries")

    # Сэмплируем
    df = df.sample(n=min(NUM_TRACKS, len(df)), random_state=42)
    print(f"Selected {len(df)} tracks")

    # Обработка
    align_dataset_parser(df, PARQUET_OUTPUT)

    # Подсчёт путей
    num_midi = df["midi_filename"].notna().sum()
    num_audio = df["audio_filename"].notna().sum()

    print(f"Кол-во треков с MIDI:  {num_midi}")
    print(f"Кол-во треков с MP3:   {num_audio}")