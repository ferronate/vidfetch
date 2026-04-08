"""
FastAPI backend for vidfetch: serve videos, auto-index, and run queries.
Run: python -m uvicorn api.main:app --reload --port 8000
"""
import time
import os
import asyncio
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Local imports — no sys.path hacks needed when running via ``python -m``
from src.video_catalog import get_video_files_cached, find_video_by_id
from src.inference import service as inference_service
from src.color_features import extract_color_summary, bhattacharyya_distance, mode_distance, passes_color_mode

DOMAIN_PROMPT_PRESETS = {
    "nature": (
        "tree, forest, mountain, hill, river, waterfall, lake, ocean, beach, sky, cloud, "
        "flower, grass, field, rock, snow, bird, fish, deer, bear"
    ),
    "food": (
        "food, meal, fruit, vegetable, plate, bowl, sandwich, pizza, burger, sushi, "
        "salad, bread, drink, cup, bottle"
    ),
    "nature-food": (
        "tree, forest, mountain, hill, river, waterfall, lake, ocean, beach, sky, cloud, "
        "flower, grass, field, rock, snow, bird, fish, deer, bear, "
        "food, meal, fruit, vegetable, plate, bowl, sandwich, pizza, burger, sushi, "
        "salad, bread, drink, cup, bottle"
    ),
}

app = FastAPI(title="Vidfetch API")

# --- CORS (fixed: the old code produced a list-inside-a-list) ---
_origins_env = os.environ.get(
    "VF_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
_allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Index helpers (delegate to inference.service for writes)
# ---------------------------------------------------------------------------

def _load_index() -> dict:
    """Load detection results from the shared index file."""
    return inference_service.load_index_snapshot()


def _save_index(data: dict):
    """Save detection results (used only by reindex to clear the file)."""
    inference_service.overwrite_index(data)


KNOWN_COLOR_FILTERS = {"any", "warm", "cool", "bright", "dark"}


def _ensure_color_summary(video_id: str, video_data: dict[str, Any]) -> Optional[dict[str, Any]]:
    summary = video_data.get("color_summary")
    if isinstance(summary, dict) and isinstance(summary.get("hue_hist"), list):
        return summary

    vpath = find_video_by_id(video_id)
    if not vpath:
        return None

    summary = extract_color_summary(vpath)
    if not summary:
        return None

    updated = dict(video_data)
    updated["color_summary"] = summary
    inference_service.upsert_index_entry(video_id, updated)
    video_data["color_summary"] = summary
    return summary


# ---------------------------------------------------------------------------
# Auto-indexing
# ---------------------------------------------------------------------------

def _submit_detection_job(
    video_path,
    detector_type: str = "auto",
    confidence: float = 0.10,
    sample_fps: float = 0.5,
    prompt: Optional[str] = None,
    save_to_index: bool = True,
    batch_size: int = 1,
) -> str:
    return inference_service.submit_job(
        str(video_path),
        detector_type=detector_type,
        confidence=confidence,
        sample_fps=sample_fps,
        prompt=prompt,
        enable_tracking=True,
        enable_adaptive_sampling=True,
        enable_temporal_aggregation=True,
        batch_size=batch_size,
        save_to_index=save_to_index,
    )


def auto_index_videos() -> list[str]:
    """Schedule indexing jobs for videos not yet in the index."""
    index = _load_index()
    videos = get_video_files_cached()

    job_ids = []
    for video_path in videos:
        video_id = video_path.stem
        if video_id in index:
            continue
        print(f"Scheduling indexing for {video_path.name}...")
        job_id = _submit_detection_job(video_path=video_path, save_to_index=True)
        job_ids.append(job_id)

    print(f"Scheduled {len(job_ids)} indexing jobs")
    return job_ids


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Schedule auto-indexing on startup (non-blocking)."""
    print("Scheduling auto-indexing in background...")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, auto_index_videos)
    loop.run_in_executor(None, inference_service.init_pool)
    loop.run_in_executor(None, lambda: inference_service.warm_workers(timeout=10.0))
    print("Auto-indexing scheduled; inference workers warming in background")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/videos")
def list_videos():
    """List all videos in data directory."""
    videos = get_video_files_cached()
    return {"videos": [{"id": f.stem, "name": f.name} for f in videos]}


@app.get("/api/video/{video_id}")
def serve_video(video_id: str):
    """Serve a video file by ID."""
    video_path = find_video_by_id(video_id)
    if not video_path:
        raise HTTPException(404, "Video not found")
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/objects")
def list_objects():
    """List detected objects from saved results."""
    index = _load_index()
    all_objects = set()
    for video_data in index.values():
        if isinstance(video_data, dict):
            all_objects.update(video_data.get("classes", []))
    return {"objects": sorted(all_objects)}


@app.get("/api/detector-info")
def get_detector_info():
    """Get information about the current detector."""
    try:
        from src.detector import VideoDetector

        detector = VideoDetector()
        return detector.get_detector_info()
    except Exception as e:
        raise HTTPException(500, f"Failed to get detector info: {e}")


@app.post("/api/query")
async def query_videos(
    object_types: Optional[List[str]] = Form(None),
    color_filter: str = Form("any"),
    k: int = Form(5),
):
    """Query videos by object types."""
    t0 = time.perf_counter()
    index = _load_index()

    types_list: List[str] = []
    if object_types:
        # Accept both repeated form fields and comma-separated values.
        for raw_value in object_types:
            for item in raw_value.split(","):
                value = item.strip()
                if value:
                    types_list.append(value)

    normalized_types = [t.lower() for t in types_list]

    color_filter_raw = (color_filter or "any").strip()
    color_mode = color_filter_raw.lower()
    reference_video_id: Optional[str] = None

    if color_mode == "same":
        # Frontend usually sends reference video id directly for "same", but handle this gracefully.
        color_mode = "any"
    elif color_mode not in KNOWN_COLOR_FILTERS:
        reference_video_id = color_filter_raw
        color_mode = "any"

    reference_summary: Optional[dict[str, Any]] = None
    if reference_video_id:
        ref_data = index.get(reference_video_id)
        if not isinstance(ref_data, dict):
            raise HTTPException(400, "Invalid color reference video id")
        reference_summary = _ensure_color_summary(reference_video_id, ref_data)
        if not reference_summary:
            raise HTTPException(400, "Reference video has no color summary yet; run detection first")

    matching_videos = []
    for video_id, video_data in index.items():
        if not isinstance(video_data, dict):
            continue

        video_classes = set(video_data.get("classes", []))
        normalized_video_classes = {str(c).lower() for c in video_classes}
        if normalized_types and not any(obj in normalized_video_classes for obj in normalized_types):
            continue

        query_object = types_list[0] if types_list else None
        query_object_normalized = normalized_types[0] if normalized_types else None
        segments = []
        if query_object_normalized:
            for entry in video_data.get("timeline", []):
                objects_in_frame = [str(d["class"]).lower() for d in entry.get("objects", []) if "class" in d]
                if query_object_normalized in objects_in_frame:
                    segments.append({"start": entry["t"], "end": entry["t"] + 0.5})

        distance = 0.1
        if reference_summary is not None:
            current_summary = _ensure_color_summary(video_id, video_data)
            if not current_summary:
                continue
            distance = bhattacharyya_distance(
                current_summary.get("hue_hist", []),
                reference_summary.get("hue_hist", []),
            )
        elif color_mode != "any":
            current_summary = _ensure_color_summary(video_id, video_data)
            if not current_summary or not passes_color_mode(current_summary, color_mode):
                continue
            distance = mode_distance(current_summary, color_mode)

        video_name = f"{video_id}.mp4"
        vpath = find_video_by_id(video_id)
        if vpath:
            video_name = vpath.name

        matching_videos.append(
            {
                "id": video_id,
                "name": video_name,
                "distance": round(float(distance), 6),
                "object_segments": segments,
            }
        )

    matching_videos.sort(key=lambda x: float(x.get("distance", 1.0)))
    matching_videos = matching_videos[:k]
    elapsed = time.perf_counter() - t0

    return {
        "results": matching_videos,
        "time_ms": round(elapsed * 1000, 2),
        "query_object": types_list[0] if types_list else None,
    }


@app.post("/api/detect")
async def run_detection(
    video_id: str,
    detector_type: str = Form("auto"),
    model: str = Form("auto"),
    oiv7: bool = Form(False),
    domain: str = Form("none"),
    prompt: Optional[str] = Form(None),
):
    """Run object detection on a specific video."""
    video_path = find_video_by_id(video_id)
    if not video_path:
        raise HTTPException(404, "Video not found")

    existing = _load_index().get(video_id)
    if isinstance(existing, dict):
        has_error = bool(existing.get("error"))
        has_timeline = isinstance(existing.get("timeline"), list) and len(existing.get("timeline", [])) > 0
        has_classes = isinstance(existing.get("classes"), list) and len(existing.get("classes", [])) > 0
        if not has_error and (has_timeline or has_classes):
            return {
                "success": True,
                "reused": True,
                "video_id": video_id,
                "message": "Reused existing indexed detections",
            }

    batch_size = int(os.environ.get("VF_BATCH_SIZE", "4"))
    from src.cpu_profile import get_cpu_profile
    profile = get_cpu_profile()
    effective_prompt = prompt
    if not effective_prompt and domain in DOMAIN_PROMPT_PRESETS:
        effective_prompt = DOMAIN_PROMPT_PRESETS[domain]

    effective_detector_type = detector_type
    if domain in DOMAIN_PROMPT_PRESETS and detector_type == "auto":
        effective_detector_type = "yolo-world"

    job_id = _submit_detection_job(
        video_path=video_path,
        detector_type=effective_detector_type,
        confidence=0.15,
        sample_fps=profile.sample_fps,
        prompt=effective_prompt,
        save_to_index=True,
        batch_size=batch_size,
    )

    return {
        "success": True,
        "reused": False,
        "job_id": job_id,
        "status_url": f"/api/detect/status/{job_id}",
        "result_url": f"/api/detect/result/{job_id}",
    }


@app.get("/api/detect/status/{job_id}")
def get_detect_status(job_id: str):
    """Get detection job status."""
    job = inference_service.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/detect/result/{job_id}")
def get_detect_result(job_id: str):
    """Get detection job result."""
    result = inference_service.get_job_result(job_id)
    if not result:
        raise HTTPException(404, "Job result not found")
    return result


@app.get("/api/detect/jobs")
def list_detect_jobs():
    """List all detection jobs."""
    return {"jobs": inference_service.list_jobs()}


# ---------------------------------------------------------------------------
# Correction endpoints
# ---------------------------------------------------------------------------

@app.get("/api/corrections/summary")
def get_corrections_summary():
    """Get correction system summary."""
    try:
        from src.correction_applier import get_correction_summary
        return get_correction_summary()
    except Exception as e:
        raise HTTPException(500, f"Failed to get corrections summary: {e}")


@app.get("/api/corrections")
def list_corrections(
    video_id: Optional[str] = None,
    class_name: Optional[str] = None,
    limit: int = 100,
):
    """List corrections with optional filters."""
    try:
        from src.corrections import get_store
        store = get_store()
        corrections = store.get_corrections(
            video_id=video_id,
            original_class=class_name,
            limit=limit,
        )
        return {
            "corrections": [
                {
                    "id": c.id,
                    "video_id": c.video_id,
                    "frame_number": c.frame_number,
                    "timestamp": c.timestamp,
                    "original_class": c.original_class,
                    "corrected_class": c.corrected_class,
                    "bbox": c.bbox,
                    "original_confidence": c.original_confidence,
                    "action": c.action,
                    "notes": c.notes,
                }
                for c in corrections
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to list corrections: {e}")


@app.post("/api/corrections")
async def add_correction(
    video_id: str = Form(...),
    frame_number: int = Form(...),
    timestamp: float = Form(0.0),
    original_class: str = Form(...),
    corrected_class: str = Form(...),
    bbox: Optional[str] = Form(None),
    original_confidence: float = Form(0.0),
    action: str = Form("relabel"),
    notes: str = Form(""),
):
    """Add a user correction."""
    try:
        from src.corrections import Correction, get_store
        import json

        store = get_store()
        correction = Correction(
            video_id=video_id,
            frame_number=frame_number,
            timestamp=timestamp,
            original_class=original_class,
            corrected_class=corrected_class,
            bbox=json.loads(bbox) if bbox else None,
            original_confidence=original_confidence,
            action=action,
            notes=notes,
        )
        correction_id = store.add_correction(correction)
        return {"success": True, "id": correction_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to add correction: {e}")


@app.delete("/api/corrections/{correction_id}")
def delete_correction(correction_id: int):
    """Delete a correction."""
    try:
        from src.corrections import get_store
        store = get_store()
        store.delete_correction(correction_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete correction: {e}")


@app.get("/api/corrections/rules")
def list_rules(enabled_only: bool = True):
    """List correction rules."""
    try:
        from src.corrections import get_store
        store = get_store()
        rules = store.get_rules(enabled_only=enabled_only)
        return {
            "rules": [
                {
                    "id": r.id,
                    "pattern_class": r.pattern_class,
                    "target_class": r.target_class,
                    "confidence_threshold": r.confidence_threshold,
                    "video_pattern": r.video_pattern,
                    "enabled": r.enabled,
                    "usage_count": r.usage_count,
                }
                for r in rules
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to list rules: {e}")


@app.post("/api/corrections/rules")
async def add_rule(
    pattern_class: str = Form(...),
    target_class: str = Form(...),
    confidence_threshold: Optional[float] = Form(None),
    video_pattern: Optional[str] = Form(None),
):
    """Add a correction rule."""
    try:
        from src.corrections import CorrectionRule, get_store
        store = get_store()
        rule = CorrectionRule(
            pattern_class=pattern_class,
            target_class=target_class,
            confidence_threshold=confidence_threshold,
            video_pattern=video_pattern,
        )
        rule_id = store.add_rule(rule)
        return {"success": True, "id": rule_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to add rule: {e}")


@app.post("/api/corrections/rules/{rule_id}/toggle")
def toggle_rule(rule_id: int, disable: bool = Form(False)):
    """Toggle a rule on/off."""
    try:
        from src.corrections import get_store
        store = get_store()
        store.toggle_rule(rule_id, not disable)
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, f"Failed to toggle rule: {e}")


@app.delete("/api/corrections/rules/{rule_id}")
def delete_rule(rule_id: int):
    """Delete a rule."""
    try:
        from src.corrections import get_store
        store = get_store()
        store.delete_rule(rule_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete rule: {e}")


@app.post("/api/corrections/generate-rules")
def generate_rules(min_occurrences: int = Form(3)):
    """Auto-generate rules from repeated corrections."""
    try:
        from src.corrections import get_store
        store = get_store()
        count = store.generate_rules_from_corrections(min_occurrences=min_occurrences)
        return {"success": True, "rules_generated": count}
    except Exception as e:
        raise HTTPException(500, f"Failed to generate rules: {e}")


@app.get("/api/video/{video_id}/detections")
def get_video_detections(video_id: str):
    """Get detection results for a specific video."""
    index = _load_index()
    video_data = index.get(video_id)
    if not video_data:
        raise HTTPException(404, "Video not found in index")

    if not isinstance(video_data, dict):
        raise HTTPException(404, "Video not found in index")

    try:
        from src.correction_applier import apply_corrections_to_timeline

        timeline = video_data.get("timeline", [])
        corrected_timeline = apply_corrections_to_timeline(timeline, video_id=video_id)

        corrected_classes = sorted(
            {
                str(obj.get("class", ""))
                for entry in corrected_timeline
                for obj in entry.get("objects", [])
                if obj.get("class")
            }
        )

        response = dict(video_data)
        response["timeline"] = corrected_timeline
        response["classes"] = corrected_classes
        response["total_classes"] = len(corrected_classes)
        response["total_detections"] = len(corrected_timeline)
        return response
    except Exception:
        # If correction application fails, return raw data rather than erroring the endpoint.
        return video_data
def detect_status(job_id: str):
    """Get status for a submitted detection job."""
    job = inference_service.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/detect/result/{job_id}")
def detect_result(job_id: str):
    """Return detection result if ready, otherwise a 202 status with progress."""
    job = inference_service.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    status = job.get("status")
    if status == "done":
        return {"success": True, "result": job.get("result")}
    elif status == "failed":
        raise HTTPException(500, f"Job failed: {job.get('error')}")
    return {"success": False, "status": status}


@app.post("/api/reindex")
async def reindex_all(detector_type: str = Form("auto")):
    """Re-index all videos."""
    t0 = time.perf_counter()
    _save_index({})
    job_ids = auto_index_videos()
    elapsed = time.perf_counter() - t0

    return {
        "success": True,
        "time_ms": round(elapsed * 1000, 2),
        "jobs_submitted": len(job_ids),
    }