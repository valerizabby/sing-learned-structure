"""
Фаза 3: Извлечение признаков: audio → bar-level chroma → SSM.

Алгоритм:
  1. Загружаем mp3 через librosa
  2. Beat tracking → массив timestamps битов
  3. Группируем по 4 бита → бары (bar timestamps)
  4. Chroma CQT, усреднённая по барам → [T_bars, 12]
  5. Cosine similarity → SSM [T_bars, T_bars]

Вход:  audio/{song_id}.mp3
Выход: features/{song_id}.pt  — {'chroma': Tensor[T,12], 'ssm': Tensor[T,T], 'bar_times': list[float]}

Запуск:
    python extract_features.py \
        --audio_dir audio/ \
        --output_dir features/
"""

import argparse
from pathlib import Path

import torch
import librosa
import numpy as np


SAMPLE_RATE = 22050
BEATS_PER_BAR = 4


def get_bar_times(y: np.ndarray, sr: int) -> np.ndarray:
    """Возвращает timestamp начала каждого бара (в секундах)."""
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # Каждые BEATS_PER_BAR битов = 1 бар
    bar_times = beat_times[::BEATS_PER_BAR]
    return bar_times


def compute_bar_chroma(y: np.ndarray, sr: int, bar_times: np.ndarray) -> np.ndarray:
    """
    Вычисляет chroma CQT и усредняет по барам.
    Возвращает [T_bars, 12].
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)  # [12, frames]
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)

    bar_chroma = []
    for i, bar_start in enumerate(bar_times):
        bar_end = bar_times[i + 1] if i + 1 < len(bar_times) else frame_times[-1] + 1e-3
        mask = (frame_times >= bar_start) & (frame_times < bar_end)
        if mask.sum() == 0:
            # Нет фреймов в баре — берём ближайший
            idx = np.argmin(np.abs(frame_times - bar_start))
            bar_chroma.append(chroma[:, idx])
        else:
            bar_chroma.append(chroma[:, mask].mean(axis=1))

    return np.stack(bar_chroma, axis=0)  # [T_bars, 12]


def chroma_to_ssm(chroma: np.ndarray) -> np.ndarray:
    """Cosine similarity матрица из chroma. Возвращает [T, T]."""
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    chroma_norm = chroma / (norms + 1e-8)
    ssm = chroma_norm @ chroma_norm.T  # [T, T]
    ssm = np.clip(ssm, 0.0, 1.0)
    return ssm.astype(np.float32)


def process_file(mp3_path: Path, output_path: Path) -> bool:
    """Обрабатывает один mp3-файл. Возвращает True при успехе."""
    try:
        y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  Load error: {e}")
        return False

    if len(y) < sr * 10:  # меньше 10 секунд — пропускаем
        print(f"  Too short ({len(y)/sr:.1f}s)")
        return False

    bar_times = get_bar_times(y, sr)
    if len(bar_times) < 8:
        print(f"  Too few bars: {len(bar_times)}")
        return False

    chroma = compute_bar_chroma(y, sr, bar_times)  # [T, 12]
    if not np.isfinite(chroma).all():
        print(f"  NaN/Inf in chroma")
        return False

    ssm = chroma_to_ssm(chroma)  # [T, T]

    torch.save({
        "chroma": torch.from_numpy(chroma),   # [T, 12]
        "ssm": torch.from_numpy(ssm),          # [T, T]
        "bar_times": bar_times.tolist(),        # list[float], длина T
        "n_bars": len(bar_times),
    }, output_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="audio/")
    parser.add_argument("--output_dir", default="features/")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(audio_dir.glob("*.mp3"))
    print(f"Found {len(mp3_files)} mp3 files")

    success, failed, skipped = 0, 0, 0

    for i, mp3_path in enumerate(mp3_files):
        song_id = mp3_path.stem
        out_path = output_dir / f"{song_id}.pt"

        if out_path.exists():
            skipped += 1
            continue

        print(f"[{i+1}/{len(mp3_files)}] {song_id} ...", end=" ", flush=True)
        ok = process_file(mp3_path, out_path)
        if ok:
            success += 1
            n_bars = torch.load(out_path)["n_bars"]
            print(f"OK ({n_bars} bars)")
        else:
            failed += 1

    print(f"\nDone: {success} ok, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    main()
