"""
FastAPI backend for vidfetch: list indexed videos, run query, serve video files.
Run: python -m uvicorn api.main:app --reload --port 8000
"""
import time
from pathlib import Path
import sys
import tempfile

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.index import VideoIndex
from src.object_index import load_object_index, search_by_object, get_object_segments, _get_classes_for_video
from src.retrieval import load_index
from src.extract import video_to_feature, get_preset_color_feature

INDEX_DIR = ROOT / "index_store"

# Object labels for dropdown (Pascal VOC, skip background)
OBJECT_LABELS = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor",
]

app = FastAPI(title="Vidfetch API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_index: VideoIndex | None = None


def get_index() -> VideoIndex:
    global _index
    if _index is None:
        if not (INDEX_DIR / "features.npy").exists():
            raise HTTPException(503, "Index not built. Run: python -m scripts.build_index data")
        _index = load_index(INDEX_DIR)
    return _index


def _get_available_object_labels() -> list[str]:
    """Labels for filter dropdown: from object index if present, else Pascal VOC."""
    obj_index = load_object_index(INDEX_DIR)
    if not obj_index:
        return OBJECT_LABELS
    seen = set(OBJECT_LABELS)
    for entry in obj_index.values():
        for c in _get_classes_for_video(entry):
            seen.add(c)
    return sorted(seen)


@app.get("/api/objects")
def list_objects():
    """List object labels for search/filter dropdown (from index or Pascal VOC)."""
    return {"objects": _get_available_object_labels()}


@app.get("/api/videos")
def list_videos():
    """List all indexed videos (id, name for display)."""
    idx = get_index()
    return {
        "videos": [
            {"id": vid_id, "name": Path(p).name}
            for vid_id, p in zip(idx.ids, idx.paths)
        ]
    }


@app.get("/api/video/{video_id}")
def serve_video(video_id: str):
    """Serve a video file by ID (path is the stored path; we resolve to file)."""
    idx = get_index()
    try:
        i = idx.ids.index(video_id)
    except ValueError:
        raise HTTPException(404, "Video not found")
    path = Path(idx.paths[i])
    if not path.exists():
        raise HTTPException(404, "Video file not found on disk")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/query")
async def run_query(
    video_id: str | None = Form(None),
    file: UploadFile | None = File(None),
    object_types: list[str] | None = Form(None),
    color_filter: str | None = Form(None),
    k: int = Form(5),
):
    """
    Run retrieval: object_types (videos containing any of these), color_filter (sort by color).
    If object_types empty, all videos. color_filter: "any" | "warm" | "cool" | "bright" | "dark" | video_id.
    """
    t0 = time.perf_counter()
    idx = get_index()
    object_index = load_object_index(INDEX_DIR)

    types_to_use = [t.strip() for t in (object_types or []) if t and t.strip()]

    candidate_ids = set(idx.ids)
    if types_to_use:
        by_object = set()
        for obj in types_to_use:
            by_object.update(search_by_object(object_index, obj))
        if not by_object and object_index:
            return {"results": [], "time_ms": round((time.perf_counter() - t0) * 1000, 2), "query_object": types_to_use[0] if types_to_use else None}
        if by_object:
            candidate_ids = by_object

    query_path: Path | None = None
    is_temp = False
    if video_id:
        try:
            i = idx.ids.index(video_id)
            query_path = Path(idx.paths[i])
        except ValueError:
            pass
    if query_path is None and file and file.filename:
        suffix = Path(file.filename).suffix or ".mp4"
        fd, path = tempfile.mkstemp(suffix=suffix)
        with open(fd, "wb") as f:
            f.write(await file.read())
        query_path = Path(path)
        is_temp = True

    if not candidate_ids:
        return {"results": [], "time_ms": round((time.perf_counter() - t0) * 1000, 2), "query_object": types_to_use[0] if types_to_use else None}

    ref_feature = None
    if query_path is not None:
        ref_feature = video_to_feature(query_path)
    elif color_filter and color_filter.strip().lower() != "any":
        cf = color_filter.strip().lower()
        if cf in ("warm", "cool", "bright", "dark"):
            ref_feature = get_preset_color_feature(cf)
        else:
            ref_feature = idx.get_feature(cf)
    if ref_feature is None:
        ref_feature = get_preset_color_feature("bright")

    try:
        results = idx.search_among(ref_feature, candidate_ids, k=k)
        elapsed = time.perf_counter() - t0
        query_object = types_to_use[0] if types_to_use else None
        out_results = []
        for r in results:
            vid_id, path, dist = r[0], r[1], r[2]
            item = {"id": vid_id, "name": Path(path).name, "distance": round(dist, 6)}
            if query_object:
                segments = get_object_segments(object_index, vid_id, query_object, frame_interval=0.5)
                item["object_segments"] = segments
            out_results.append(item)
        return {
            "results": out_results,
            "time_ms": round(elapsed * 1000, 2),
            "query_object": query_object,
        }
    finally:
        if is_temp and query_path and query_path.exists():
            query_path.unlink(missing_ok=True)
