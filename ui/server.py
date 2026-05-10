"""
FastAPI backend для UI генерации музыки.

Запуск:
    cd /path/to/sing-learned-structure
    uvicorn ui.server:app --reload --port 8000
"""

import asyncio
import io
import json
import os
import threading
import traceback
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    HTMLResponse, FileResponse, StreamingResponse, JSONResponse
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from Seg2SSM.affinity_ssm import AffinitySSM, LABEL_NAMES
from pipeline.generate import (
    parse_segment_plan, build_ssm, run_pipeline, SSMType, BEATS_PER_BAR
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Music Generation UI")

STATIC_DIR = Path(__file__).parent / "static"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "ui_jobs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Job store ────────────────────────────────────────────────────────────────

# job_id → {status, progress, message, out_dir, error}
_jobs: Dict[str, Dict[str, Any]] = {}
_job_events: Dict[str, threading.Event] = {}


def _new_job() -> str:
    jid = str(uuid.uuid4())[:8]
    _jobs[jid] = {
        "status": "pending",   # pending | running | done | error
        "step": 0,
        "total_steps": 5,
        "message": "Waiting...",
        "out_dir": str(OUTPUTS_DIR / jid),
        "error": None,
    }
    _job_events[jid] = threading.Event()
    return jid


def _update_job(jid: str, step: int, message: str, status: str = "running"):
    if jid in _jobs:
        _jobs[jid]["step"] = step
        _jobs[jid]["message"] = message
        _jobs[jid]["status"] = status
        # Notify SSE listeners
        if jid in _job_events:
            _job_events[jid].set()
            _job_events[jid].clear()


# ── Schemas ──────────────────────────────────────────────────────────────────

class SegmentItem(BaseModel):
    label: str   # intro, verse, chorus, bridge, instr, outro, other
    bars: int


class GenerateRequest(BaseModel):
    prompt: str
    segments: List[SegmentItem]
    tempo: float = 120.0
    n_prefix_bars: int = 8
    ssm_type: str = "affinity"
    model_path: str


class SSMPreviewRequest(BaseModel):
    segments: List[SegmentItem]
    ssm_size: int = 256


# ── Helpers ──────────────────────────────────────────────────────────────────

LABEL_NAME_TO_ID = {v: k for k, v in LABEL_NAMES.items()}


def _segments_to_plan(segments: List[SegmentItem]):
    """SegmentItem list → [(label_id, n_bars), ...]"""
    plan = []
    for s in segments:
        label = s.label.lower()
        if label not in LABEL_NAME_TO_ID:
            raise ValueError(f"Unknown label: {label}")
        plan.append((LABEL_NAME_TO_ID[label], s.bars))
    return plan


def _build_ssm_png(segments: List[SegmentItem], ssm_size: int = 256) -> bytes:
    """Строит SSM и возвращает PNG как bytes."""
    plan = _segments_to_plan(segments)
    ssm = AffinitySSM.fixed().build(plan, ssm_size=ssm_size)
    ssm_np = ssm.numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(ssm_np, aspect="auto", origin="lower",
                   cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Границы секций
    cursor = 0
    label_positions = []
    for label_id, n_bars in plan:
        mid = cursor + n_bars * BEATS_PER_BAR / 2
        label_positions.append((mid, LABEL_NAMES.get(label_id, "?")))
        cursor += n_bars * BEATS_PER_BAR
        if cursor < ssm_size:
            ax.axvline(cursor - 0.5, color="white", lw=0.8, alpha=0.7)
            ax.axhline(cursor - 0.5, color="white", lw=0.8, alpha=0.7)

    ax.set_title("Affinity SSM", fontsize=12)
    ax.set_xlabel("Beat")
    ax.set_ylabel("Beat")

    # Подписи секций по оси X
    if label_positions:
        xs = [p[0] for p in label_positions]
        lbls = [p[1] for p in label_positions]
        ax.set_xticks(xs)
        ax.set_xticklabels(lbls, rotation=45, fontsize=7)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _run_pipeline_thread(jid: str, req: GenerateRequest):
    """Запускает пайплайн в отдельном потоке, обновляя прогресс."""
    out_dir = _jobs[jid]["out_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        _update_job(jid, 1, "Загрузка Text2Prefix...")

        try:
            from Text2Prefix.text2prefix_midi import Text2PrefixMIDI as Text2Prefix
            _backend = "text2midi"
        except ImportError:
            from Text2Prefix.text2prefix import Text2Prefix  # type: ignore[assignment]
            _backend = "musicgen+basicpitch"

        from pipeline.generate import (
            build_ssm, generate_from_prefix, piano_roll_to_midi,
            save_visualizations
        )
        from SingLS.config.config import DEVICE

        plan = _segments_to_plan(req.segments)
        ssm_type = SSMType(req.ssm_type)

        # Длина генерации = сумма всех баров - prefix
        total_bars = sum(s.bars for s in req.segments)
        T_gen = max(total_bars * BEATS_PER_BAR - req.n_prefix_bars * BEATS_PER_BAR, 0)

        # Step 1: Text → Prefix
        _update_job(jid, 1, f"Генерация музыкального префикса ({_backend})...")
        text2prefix = Text2Prefix()
        prefix_roll, detected_tempo, num_beats = text2prefix.generate(
            prompt=req.prompt,
            n_bars=req.n_prefix_bars,
            tempo=req.tempo,
        )
        # Реальный T_prefix из shape — text2midi может сгенерировать короче бюджета
        T_prefix = prefix_roll.shape[0]
        T_total = T_prefix + T_gen

        # Step 2: SSM
        _update_job(jid, 2, "Построение SSM из сегментного плана...")
        ssm = build_ssm(ssm_type, plan, T_total)
        ssm = ssm.to(DEVICE)

        # Step 3: Load model
        _update_job(jid, 3, "Загрузка модели SingLS...")
        model = torch.load(req.model_path, weights_only=False, map_location=DEVICE)
        model.eval()

        # Step 4: Generate
        _update_job(jid, 4, f"Авторегрессивная генерация ({T_gen} битов)...")
        sequence = generate_from_prefix(model, prefix_roll, ssm, T_gen)
        full_roll = sequence.squeeze(1).detach().cpu().numpy().round()
        full_roll = full_roll[:T_total]

        # Step 5: Export
        _update_job(jid, 5, "Сохранение MIDI и визуализаций...")
        out_path = Path(out_dir)

        # Сохраняем только сгенерированную часть (без prefix)
        midi = piano_roll_to_midi(full_roll[T_prefix:], detected_tempo)
        midi.write(str(out_path / "generated.mid"))

        save_visualizations(
            out_path, prefix_roll, full_roll, ssm.cpu(),
            detected_tempo, ssm_type.value, plan,
        )

        _jobs[jid]["status"] = "done"
        _jobs[jid]["step"] = 5
        _jobs[jid]["message"] = "Готово!"
        if jid in _job_events:
            _job_events[jid].set()

    except Exception as e:
        _jobs[jid]["status"] = "error"
        _jobs[jid]["error"] = str(e)
        _jobs[jid]["message"] = f"Ошибка: {e}"
        tb = traceback.format_exc()
        print(f"[Job {jid}] ERROR:\n{tb}")
        if jid in _job_events:
            _job_events[jid].set()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return html_path.read_text()


@app.get("/api/models")
async def list_models():
    """Возвращает список доступных чекпоинтов моделей."""
    root = Path(__file__).parent.parent
    models = []

    # Основные модели: data/meta_info/*/model_30_epochs.txt
    meta_info = root / "data" / "meta_info"
    if meta_info.exists():
        for txt in sorted(meta_info.glob("*/model_30_epochs.txt")):
            models.append(str(txt.relative_to(root)))

    # Дополнительно: .pt файлы (не в venv/data/features)
    skip = {"venv", "data/", "features", "Seg2SSM/data_prep"}
    for pt in sorted(root.rglob("*.pt")):
        rel = str(pt.relative_to(root))
        if any(s in rel for s in skip):
            continue
        models.append(rel)

    return {"models": models}


@app.post("/api/ssm-preview")
async def ssm_preview(req: SSMPreviewRequest):
    """Быстрый превью SSM в PNG (без модели)."""
    if not req.segments:
        raise HTTPException(400, "segments is empty")
    try:
        png = _build_ssm_png(req.segments, req.ssm_size)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Запускает генерацию в background, возвращает job_id."""
    if not req.segments:
        raise HTTPException(400, "segments is empty")
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is empty")
    model_file = Path(req.model_path)
    # Путь может быть абсолютным или относительным от корня проекта
    if not model_file.is_absolute():
        model_file = Path(__file__).parent.parent / model_file
    if not model_file.exists():
        raise HTTPException(400, f"Model not found: {req.model_path}")
    req.model_path = str(model_file)

    jid = _new_job()
    t = threading.Thread(target=_run_pipeline_thread, args=(jid, req), daemon=True)
    t.start()
    return {"job_id": jid}


@app.get("/api/progress/{job_id}")
async def progress(job_id: str):
    """SSE стрим прогресса генерации."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    async def event_stream():
        last_step = -1
        while True:
            job = _jobs.get(job_id, {})
            step = job.get("step", 0)
            status = job.get("status", "pending")
            message = job.get("message", "")
            total = job.get("total_steps", 5)

            if step != last_step or status in ("done", "error"):
                data = json.dumps({
                    "step": step,
                    "total": total,
                    "message": message,
                    "status": status,
                    "error": job.get("error"),
                })
                yield f"data: {data}\n\n"
                last_step = step

            if status in ("done", "error"):
                break

            # Ждём следующего обновления (или таймаут 1 сек для heartbeat)
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    result = dict(job)
    if job["status"] == "done":
        result["midi_url"] = f"/api/midi/{job_id}"
        result["ssm_url"] = f"/api/ssm-result/{job_id}"
    return result


@app.get("/api/midi/{job_id}")
async def get_midi(job_id: str):
    """Отдаёт сгенерированный MIDI файл (для player и download)."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    midi_path = Path(job["out_dir"]) / "generated.mid"
    if not midi_path.exists():
        raise HTTPException(404, "MIDI not ready")
    return FileResponse(
        str(midi_path),
        media_type="audio/midi",
        filename="generated.mid",
    )


@app.get("/api/download/{job_id}")
async def download_midi(job_id: str):
    """Скачать MIDI с Content-Disposition: attachment."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    midi_path = Path(job["out_dir"]) / "generated.mid"
    if not midi_path.exists():
        raise HTTPException(404, "MIDI not ready")
    return FileResponse(
        str(midi_path),
        media_type="audio/midi",
        filename="generated.mid",
        headers={"Content-Disposition": "attachment; filename=generated.mid"},
    )


@app.get("/api/ssm-result/{job_id}")
async def ssm_result(job_id: str):
    """Отдаёт PNG с обзором пайплайна (SSM + piano roll)."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    img_path = Path(job["out_dir"]) / "pipeline_overview.png"
    if not img_path.exists():
        raise HTTPException(404, "Image not ready")
    return FileResponse(str(img_path), media_type="image/png")
