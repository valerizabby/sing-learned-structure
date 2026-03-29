"""
Фаза 1: Парсинг SALAMI аннотаций.

Формат textfile1.txt:
  0.000   Silence
  0.464   A, Intro, (piano, b       ← новая секция: uppercase + функциональная метка
  5.191   b                          ← продолжение предыдущей секции (только lowercase)
  33.62   C, Verse, (vocal, e       ← новая секция

Логика: секция начинается на строке с заглавной буквой или функциональной меткой.
        Строки только с малыми буквами и инструментами — продолжение.

Источник данных: Internet Archive (IA) — 446 треков с прямыми ссылками для скачивания.
Треки Codaich/Isophonics требуют YouTube и не используются.

Запуск:
    python parse_annotations.py \
        --salami_dir /path/to/salami-data-public \
        --source_filter IA \
        --class_filter all \
        --output annotations.json
"""

import argparse
import json
import re
import csv
from pathlib import Path
from typing import Optional, List, Dict


# Функциональные метки → нормализованный класс
FUNCTIONAL_LABELS = {
    "silence": None,
    "noise": None,
    "applause": None,
    "end": None,

    "intro": "intro",
    "introduction": "intro",

    "verse": "verse",
    "pre-verse": "verse",
    "verse_1": "verse",
    "verse_2": "verse",

    "chorus": "chorus",
    "hook": "chorus",
    "refrain": "chorus",
    "pre-chorus": "chorus",

    "bridge": "bridge",
    "transition": "bridge",
    "interlude": "bridge",

    "solo": "instrumental",
    "instrumental": "instrumental",
    "break": "instrumental",

    "outro": "outro",
    "coda": "outro",
    "fade-out": "outro",
    "fade_out": "outro",
    "ending": "outro",
}

LABEL_VOCAB = {
    "intro": 0,
    "verse": 1,
    "chorus": 2,
    "bridge": 3,
    "instrumental": 4,
    "outro": 5,
    "other": 6,
}


def extract_functional_label(line_label: str) -> Optional[str]:
    """
    Из строки вида "A, Intro, (piano, b" извлекает функциональную метку.
    Возвращает нормализованную метку или None если метки нет / это silence.
    """
    # Разбиваем по запятым, убираем скобочные выражения
    parts = re.split(r"[,()]", line_label)
    for part in parts:
        token = part.strip().lower()
        token = re.sub(r"[^a-z0-9_\-]", "", token)
        if token in FUNCTIONAL_LABELS:
            return FUNCTIONAL_LABELS[token]
    return None


def is_section_boundary(line_label: str) -> bool:
    """
    Строка является началом новой секции если содержит заглавную букву
    или явную функциональную метку (не просто продолжение типа "b", "c, guitar)").
    """
    parts = re.split(r"[,()]", line_label)
    for part in parts:
        token = part.strip()
        # Заглавная буква сама по себе (A, B, C', V, C, I и т.д.)
        if re.match(r"^[A-Z]['\']?$", token):
            return True
        # Явная функциональная метка
        if token.lower().rstrip("_0123456789") in FUNCTIONAL_LABELS:
            return True
    return False


def parse_textfile(path: Path) -> List[dict]:
    """
    Парсит textfile1.txt.
    Возвращает только строки-границы секций: {'start_sec', 'raw_label'}.
    """
    boundaries = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            try:
                start = float(parts[0])
            except ValueError:
                continue
            raw_label = parts[1].strip()

            if is_section_boundary(raw_label):
                boundaries.append({"start_sec": start, "raw_label": raw_label})

    return boundaries


def boundaries_to_segments(boundaries: List[dict], total_duration: float) -> Optional[List[dict]]:
    """
    Превращает границы в сегменты с меткой, start_sec, end_sec.
    Фильтрует silence/None. Требует минимум 2 разных функциональных класса.
    """
    segments = []
    for i, b in enumerate(boundaries):
        label = extract_functional_label(b["raw_label"])
        if label is None:
            continue
        start = b["start_sec"]
        # Конец = начало следующей границы (любой, не только функциональной)
        end = boundaries[i + 1]["start_sec"] if i + 1 < len(boundaries) else total_duration
        if end - start < 1.0:
            continue
        segments.append({
            "label": label,
            "label_id": LABEL_VOCAB.get(label, LABEL_VOCAB["other"]),
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(end - start, 3),
        })

    if not segments:
        return None
    unique = {s["label"] for s in segments}
    if len(unique) < 2:
        return None
    return segments


def load_metadata(salami_dir: Path) -> Dict[str, dict]:
    """Загружает metadata/metadata.csv → {song_id: {...}}."""
    meta_path = salami_dir / "metadata" / "metadata.csv"
    meta = {}
    if not meta_path.exists():
        print(f"WARNING: metadata not found at {meta_path}")
        return meta
    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = row.get("SONG_ID", "").strip()
            if not song_id:
                continue
            discarded = row.get("SONG_WAS_DISCARDED_FLAG", "FALSE").strip().upper()
            if discarded == "TRUE":
                continue
            meta[song_id] = {
                "title": row.get("SONG_TITLE", "").strip(),
                "artist": row.get("ARTIST", "").strip(),
                "duration": float(row.get("SONG_DURATION", 0) or 0),
                "source": row.get("SOURCE", "").strip(),
                "class": row.get("CLASS", "").strip(),
                "genre": row.get("GENRE", "").strip(),
            }
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--salami_dir", required=True)
    parser.add_argument("--output", default="annotations.json")
    parser.add_argument("--source_filter", default="IA",
                        help="Фильтр по SOURCE в metadata (IA / Codaich / Isophonics / all)")
    parser.add_argument("--class_filter", default="all",
                        help="Фильтр по CLASS (popular / Live_Music_Archive / all)")
    args = parser.parse_args()

    salami_dir = Path(args.salami_dir)
    ann_dir = salami_dir / "annotations"
    meta = load_metadata(salami_dir)
    print(f"Loaded metadata: {len(meta)} tracks")

    results = []
    skipped_no_meta = skipped_filter = skipped_parse = 0

    for song_dir in sorted(ann_dir.iterdir()):
        if not song_dir.is_dir():
            continue
        song_id = song_dir.name
        textfile = song_dir / "textfile1.txt"
        if not textfile.exists():
            skipped_no_meta += 1
            continue

        info = meta.get(song_id)
        if info is None:
            skipped_no_meta += 1
            continue

        # Фильтр по источнику
        if args.source_filter != "all" and info["source"] != args.source_filter:
            skipped_filter += 1
            continue

        # Фильтр по классу
        if args.class_filter != "all" and info["class"] != args.class_filter:
            skipped_filter += 1
            continue

        boundaries = parse_textfile(textfile)
        if not boundaries:
            skipped_parse += 1
            continue

        duration = info["duration"] or (boundaries[-1]["start_sec"] + 30)
        segments = boundaries_to_segments(boundaries, duration)
        if segments is None:
            skipped_parse += 1
            continue

        results.append({
            "song_id": song_id,
            "title": info["title"],
            "artist": info["artist"],
            "duration_sec": duration,
            "source": info["source"],
            "genre": info["genre"],
            "segments": segments,
        })

    print(f"Parsed: {len(results)} tracks")
    print(f"Skipped: no_meta={skipped_no_meta}, filter={skipped_filter}, parse={skipped_parse}")

    if results:
        from collections import Counter
        label_counts = Counter(s["label"] for r in results for s in r["segments"])
        print("Label distribution:")
        for label, count in label_counts.most_common():
            print(f"  {label}: {count}")

    with open(args.output, "w") as f:
        json.dump({"label_vocab": LABEL_VOCAB, "tracks": results}, f, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
