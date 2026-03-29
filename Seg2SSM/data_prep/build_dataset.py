"""
Фаза 4: Сборка датасета + аугментация сегментных границ.

Алгоритм на каждый трек:
  1. Читаем segments из annotations.json
  2. Читаем bar_times + ssm из features/{song_id}.pt
  3. Выравниваем сегментные границы (секунды → бары)
  4. Строим аналитическую блочную SSM из segment_plan
  5. Аугментация:
     - boundary jitter: ±JITTER_BARS случайный сдвиг внутренних границ (N_JITTER раз)
     - time flip: переворот порядка баров (×2 к каждому варианту)
     Итого: (1 + N_JITTER) × 2 = ~14 примеров на трек
  6. Фильтрация + сохранение

Выход: seg2ssm_train.pt, seg2ssm_val.pt, seg2ssm_test.pt
       Каждый = list of {'segment_plan', 'block_ssm', 'gt_ssm', 'T', 'song_id'}

Запуск:
    python build_dataset.py \
        --annotations annotations.json \
        --features_dir features/ \
        --output_dir ../../data/seg2ssm/
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import numpy as np


N_JITTER = 6      # Количество jitter-аугментаций на трек (+ 1 оригинал)
JITTER_BARS = 2   # Максимальный сдвиг границ в барах (±)
SSM_SIZE = 64     # Фиксированный размер SSM после resize
MIN_BARS = 16


def align_segments_to_bars(segments: List[dict], bar_times: List[float]) -> List[dict]:
    """
    Переводит сегменты из секунд в бары.
    Возвращает список {'label_id', 'start_bar', 'end_bar', 'duration_bars'}.
    """
    bar_times = np.array(bar_times)
    aligned = []

    for seg in segments:
        start_bar = int(np.searchsorted(bar_times, seg["start_sec"], side="left"))
        end_bar = int(np.searchsorted(bar_times, seg["end_sec"], side="left"))
        end_bar = min(end_bar, len(bar_times))
        duration = end_bar - start_bar
        if duration < 1:
            continue
        aligned.append({
            "label": seg["label"],
            "label_id": seg["label_id"],
            "start_bar": start_bar,
            "end_bar": end_bar,
            "duration_bars": duration,
        })

    return aligned


def build_block_ssm(aligned_segments: List[dict], T: int) -> torch.Tensor:
    """
    Строит идеальную блочную SSM из сегментного плана.
    block_ssm[i,j] = 1.0 если бары i и j принадлежат секциям одного типа.
    """
    bar_labels = torch.full((T,), fill_value=-1, dtype=torch.long)
    for seg in aligned_segments:
        s, e = seg["start_bar"], min(seg["end_bar"], T)
        bar_labels[s:e] = seg["label_id"]

    # Бары без метки (gaps) → 7 (unknown)
    bar_labels[bar_labels == -1] = 7

    block_ssm = (bar_labels.unsqueeze(0) == bar_labels.unsqueeze(1)).float()
    return block_ssm  # [T, T]


def jitter_boundaries(aligned_segments: List[dict], T: int, max_jitter: int, rng: np.random.Generator) -> List[dict]:
    """
    Сдвигает внутренние границы сегментов на случайное значение в диапазоне [-max_jitter, max_jitter].
    Первая и последняя границы (начало и конец трека) не трогаются.
    Возвращает новый список сегментов с теми же label_id.
    """
    if len(aligned_segments) < 2:
        return aligned_segments

    # Внутренние границы: конец i-го = начало (i+1)-го сегмента
    boundaries = [seg["start_bar"] for seg in aligned_segments[1:]]
    new_boundaries = []
    prev = aligned_segments[0]["start_bar"] + 1  # нижняя граница для следующей точки

    for i, b in enumerate(boundaries):
        jitter = int(rng.integers(-max_jitter, max_jitter + 1))
        upper = aligned_segments[i + 1]["end_bar"] - 1  # не уходим за конец следующего сегмента
        new_b = int(np.clip(b + jitter, prev, upper))
        new_boundaries.append(new_b)
        prev = new_b + 1

    starts = [aligned_segments[0]["start_bar"]] + new_boundaries
    ends = new_boundaries + [aligned_segments[-1]["end_bar"]]

    result = []
    for i, seg in enumerate(aligned_segments):
        start, end = starts[i], ends[i]
        duration = end - start
        if duration < 1:
            continue
        result.append({**seg, "start_bar": start, "end_bar": end, "duration_bars": duration})
    return result


def flip_ssm(ssm: torch.Tensor) -> torch.Tensor:
    """Переворот порядка баров: [T,T] → [T,T] с обратным порядком строк и столбцов."""
    return ssm.flip(0).flip(1)


def chroma_to_ssm(chroma: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity из chroma [T, 12] → SSM [T, T].
    Tanh-нормализация (z-score → tanh → [0,1]):
      - z-score центрирует по среднему и масштабирует по std
      - tanh нелинейно "растягивает" контраст вблизи среднего
      - результат имеет mean ≈ 0.5 и чёткие блочные границы
    Линейная нормализация не помогает: chroma-SSM имеет сильный перекос вправо
    даже после min-max (mean ≈ 0.83 для тональной музыки).
    """
    norms = chroma.norm(dim=1, keepdim=True)
    chroma_norm = chroma / (norms + 1e-8)
    ssm = torch.matmul(chroma_norm, chroma_norm.T).clamp(0.0, 1.0)
    z = (ssm - ssm.mean()) / (ssm.std() + 1e-8)   # z-score
    ssm = (torch.tanh(z * 2) + 1) / 2              # → [0,1], mean ≈ 0.5
    return ssm


def resize_ssm(ssm: torch.Tensor, size: int) -> torch.Tensor:
    """Resize [T,T] → [size,size] через bilinear интерполяцию."""
    return torch.nn.functional.interpolate(
        ssm.unsqueeze(0).unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def make_segment_plan(aligned_segments: List[dict]) -> List[Tuple[int, int]]:
    """Возвращает [(label_id, duration_bars), ...] — сам по себе план."""
    return [(s["label_id"], s["duration_bars"]) for s in aligned_segments]


def process_track(track: dict, features: dict, rng: np.random.Generator) -> Optional[List[dict]]:
    """
    Собирает все аугментированные примеры для одного трека.
    Аугментации:
      - оригинал
      - N_JITTER версий с boundary jitter
      - каждый из вариантов ещё раз с time flip
    Возвращает None если трек не прошёл фильтрацию.
    """
    bar_times = features["bar_times"]
    chroma = features["chroma"]   # [T_raw, 12]
    T_raw = chroma.shape[0]

    if T_raw < MIN_BARS:
        return None

    aligned = align_segments_to_bars(track["segments"], bar_times)
    if not aligned:
        return None

    unique_labels = {s["label_id"] for s in aligned}
    if len(unique_labels) < 2:
        return None

    # Обрезаем до T_raw (на случай расхождения разметки и аудио)
    for seg in aligned:
        seg["end_bar"] = min(seg["end_bar"], T_raw)
    aligned = [s for s in aligned if s["duration_bars"] > 0]
    if not aligned:
        return None

    gt_ssm_raw = chroma_to_ssm(chroma)  # [T_raw, T_raw] — не зависит от аугментации
    if not torch.isfinite(gt_ssm_raw).all():
        return None

    gt_ssm = resize_ssm(gt_ssm_raw, SSM_SIZE)       # [SSM_SIZE, SSM_SIZE]
    gt_ssm_flipped = flip_ssm(gt_ssm)               # time flip gt

    # Генерируем варианты сегментных планов: оригинал + N_JITTER jitter-версий
    segment_variants = [aligned]
    for _ in range(N_JITTER):
        jittered = jitter_boundaries(aligned, T_raw, JITTER_BARS, rng)
        if jittered:
            segment_variants.append(jittered)

    examples = []
    for aug_idx, segs in enumerate(segment_variants):
        block_ssm_raw = build_block_ssm(segs, T_raw)        # [T_raw, T_raw]
        block_ssm = resize_ssm(block_ssm_raw, SSM_SIZE)      # [SSM_SIZE, SSM_SIZE]
        segment_plan = make_segment_plan(segs)

        # Оригинальный порядок времени
        examples.append({
            "song_id": track["song_id"],
            "aug": f"jitter_{aug_idx}",
            "segment_plan": segment_plan,
            "block_ssm": block_ssm,          # [SSM_SIZE, SSM_SIZE]
            "gt_ssm": gt_ssm,                # [SSM_SIZE, SSM_SIZE]
            "T_original": T_raw,
        })

        # Time flip: тот же трек в обратном порядке баров
        block_ssm_f = flip_ssm(block_ssm)
        examples.append({
            "song_id": track["song_id"],
            "aug": f"jitter_{aug_idx}_flip",
            "segment_plan": [(lid, dur) for lid, dur in reversed(segment_plan)],
            "block_ssm": block_ssm_f,
            "gt_ssm": gt_ssm_flipped,
            "T_original": T_raw,
        })

    return examples if examples else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--features_dir", default="features/")
    parser.add_argument("--output_dir", default="../../data/seg2ssm/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotations) as f:
        data = json.load(f)

    tracks = data["tracks"]
    print(f"Tracks in annotations: {len(tracks)}")

    all_examples = []
    processed, failed = 0, 0

    for track in tracks:
        feat_path = features_dir / f"{track['song_id']}.pt"
        if not feat_path.exists():
            failed += 1
            continue

        features = torch.load(feat_path, weights_only=True)
        examples = process_track(track, features, rng)
        if examples is None:
            failed += 1
            continue

        all_examples.extend(examples)
        processed += 1

    print(f"Processed: {processed} tracks → {len(all_examples)} examples "
          f"(x{len(all_examples)//max(processed,1)} aug, strategy: jitter×{N_JITTER}+flip), failed: {failed}")

    # Train/val/test split по трекам (не по примерам — чтобы не было лика)
    unique_songs = list({e["song_id"] for e in all_examples})
    random.shuffle(unique_songs)
    n = len(unique_songs)
    train_songs = set(unique_songs[:int(n * 0.70)])
    val_songs   = set(unique_songs[int(n * 0.70):int(n * 0.85)])
    test_songs  = set(unique_songs[int(n * 0.85):])

    train = [e for e in all_examples if e["song_id"] in train_songs]
    val   = [e for e in all_examples if e["song_id"] in val_songs]
    test  = [e for e in all_examples if e["song_id"] in test_songs]

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    torch.save({"label_vocab": data["label_vocab"], "examples": train},
               output_dir / "seg2ssm_train.pt")
    torch.save({"label_vocab": data["label_vocab"], "examples": val},
               output_dir / "seg2ssm_val.pt")
    torch.save({"label_vocab": data["label_vocab"], "examples": test},
               output_dir / "seg2ssm_test.pt")

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
