"""
Фаза 2: Скачивание аудио с Internet Archive (прямые HTTP-ссылки).

Не требует YouTube, yt-dlp или cookies.
Требует: pip install requests

Вход:  annotations.json (из parse_annotations.py, source_filter=IA)
       id_index_internetarchive.csv (из salami-data-public/metadata/)
Выход: audio/{song_id}.mp3

Логика: URL из CSV (~2012) могут быть устаревшими. Скрипт сначала пробует
        оригинальный URL, при 404 — запрашивает metadata API archive.org
        и ищет актуальный аудиофайл в том же item.

Запуск:
    python download_audio.py \
        --annotations annotations.json \
        --ia_csv /path/to/salami-data-public/metadata/id_index_internetarchive.csv \
        --output_dir audio/
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

AUDIO_EXTS = (".mp3", ".ogg", ".flac", ".m4a", ".wav")
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "SALAMI-research/1.0 (academic)"


def load_ia_mapping(csv_path: Path) -> dict:
    """Загружает маппинг song_id → URL из id_index_internetarchive.csv."""
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            song_id = row.get("SONG_ID", "").strip()
            url = row.get("URL", "").strip()
            if song_id and url:
                mapping[song_id] = url
    return mapping


def resolve_url(original_url: str) -> Optional[str]:
    """
    Возвращает актуальный URL для скачивания.
    Сначала проверяет HEAD на оригинальный URL, при 404 ищет через metadata API.
    """
    # Нормализуем http → https
    url = original_url.replace("http://", "https://", 1)

    # Быстрая проверка: HEAD-запрос
    try:
        r = SESSION.head(url, timeout=15, allow_redirects=True)
        if r.status_code == 200:
            return url
    except requests.exceptions.RequestException:
        pass

    # Извлекаем identifier и имя файла из URL
    # Формат: https://archive.org/download/{identifier}/{filename}
    m = re.search(r"archive\.org/download/([^/]+)/(.+)", url)
    if not m:
        return None
    identifier, original_name = m.group(1), m.group(2)

    # Запрашиваем metadata API
    try:
        meta_resp = SESSION.get(
            f"https://archive.org/metadata/{identifier}", timeout=20
        )
        if meta_resp.status_code != 200:
            return None
        files = meta_resp.json().get("files", [])
    except requests.exceptions.RequestException:
        return None

    audio_files = [
        f["name"] for f in files
        if f.get("name", "").lower().endswith(AUDIO_EXTS)
        and f.get("source", "") != "metadata"
    ]
    if not audio_files:
        return None

    # Предпочитаем файл максимально похожий на оригинальное имя
    orig_lower = original_name.lower()
    for name in audio_files:
        if name.lower() == orig_lower:
            return f"https://archive.org/download/{identifier}/{name}"

    # Иначе — первый mp3 (или любой аудиофайл)
    mp3s = [n for n in audio_files if n.lower().endswith(".mp3")]
    chosen = mp3s[0] if mp3s else audio_files[0]
    return f"https://archive.org/download/{identifier}/{chosen}"


def download_track(url: str, output_path: Path, timeout: int = 90) -> bool:
    """Скачивает аудиофайл по URL. Возвращает True при успехе."""
    try:
        response = SESSION.get(url, stream=True, timeout=timeout,
                               allow_redirects=True)
        if response.status_code != 200:
            print(f"  [HTTP {response.status_code}]", end=" ")
            return False
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        if output_path.stat().st_size < 1024:
            output_path.unlink()
            print("  [empty file]", end=" ")
            return False
        return True
    except requests.exceptions.Timeout:
        print("  [timeout]", end=" ")
    except requests.exceptions.RequestException as e:
        print(f"  [{e}]", end=" ")
    if output_path.exists():
        output_path.unlink()
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--ia_csv", required=True,
                        help="Путь к id_index_internetarchive.csv")
    parser.add_argument("--output_dir", default="audio/")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Пауза между запросами (сек)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Скачать только N треков (0 = все)")
    parser.add_argument("--timeout", type=int, default=90,
                        help="Таймаут на один файл (сек)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotations) as f:
        data = json.load(f)
    track_ids = {t["song_id"] for t in data["tracks"]}

    ia_map = load_ia_mapping(Path(args.ia_csv))
    matched = track_ids & set(ia_map)
    print(f"Tracks in annotations: {len(track_ids)}")
    print(f"Tracks with IA URL:    {len(matched)}")

    to_download = [(sid, ia_map[sid]) for sid in sorted(matched)]
    if args.limit:
        to_download = to_download[: args.limit]

    success, failed, skipped = 0, 0, 0

    for i, (song_id, original_url) in enumerate(to_download):
        out_path = output_dir / f"{song_id}.mp3"
        if out_path.exists():
            skipped += 1
            continue

        print(f"[{i+1}/{len(to_download)}] {song_id} ...", end=" ", flush=True)

        url = resolve_url(original_url)
        if url is None:
            failed += 1
            print("FAIL  [item not found on archive.org]")
            time.sleep(args.delay)
            continue

        ok = download_track(url, out_path, timeout=args.timeout)
        if ok:
            success += 1
            print("OK")
        else:
            failed += 1
            print("FAIL")

        time.sleep(args.delay)

    print(f"\nDone: {success} ok, {failed} failed, {skipped} already exist")
    print(f"Audio files in {output_dir}: {len(list(output_dir.glob('*.mp3')))}")


if __name__ == "__main__":
    main()
