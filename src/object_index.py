"""
Object index: video_id -> classes and per-frame timeline. Load/save objects.json.
Supports both legacy format { "video_id": ["class1", ...] } and
new format { "video_id": { "classes": [...], "frames": [ {"t": 0.0, "objects": [...]}, ... ] } }.
"""
from pathlib import Path
import json


def load_object_index(index_dir: str | Path) -> dict:
    """Load object index from index_dir/objects.json. Values may be list (classes) or dict (classes + frames)."""
    path = Path(index_dir) / "objects.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _get_classes_for_video(entry) -> list[str]:
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict) and "classes" in entry:
        return entry["classes"]
    return []


def search_by_object(
    object_index: dict,
    object_label: str,
) -> list[str]:
    """Return list of video_ids that contain the given object (case-insensitive)."""
    label_lower = object_label.strip().lower()
    if not label_lower:
        return []
    return [
        vid_id for vid_id, entry in object_index.items()
        if label_lower in [c.lower() for c in _get_classes_for_video(entry)]
    ]


def _merge_adjacent_segments(times_with_object: list[float], frame_interval: float) -> list[dict]:
    """Merge consecutive timestamps into segments {start, end}. Assume each t is a sample; segment spans t to t+frame_interval."""
    if not times_with_object:
        return []
    sorted_t = sorted(times_with_object)
    segments = []
    start = sorted_t[0]
    end = start + frame_interval
    for t in sorted_t[1:]:
        if t <= end + frame_interval * 0.5:
            end = t + frame_interval
        else:
            segments.append({"start": round(start, 2), "end": round(end, 2)})
            start = t
            end = t + frame_interval
    segments.append({"start": round(start, 2), "end": round(end, 2)})
    return segments


def get_object_segments(
    object_index: dict,
    video_id: str,
    object_label: str,
    frame_interval: float = 0.5,
) -> list[dict]:
    """
    Return list of {start, end} segments (seconds) where object_label appears in video_id.
    Returns [] if no timeline data or object not in video.
    """
    entry = object_index.get(video_id)
    if not entry or not isinstance(entry, dict) or "frames" not in entry:
        return []
    label_lower = object_label.strip().lower()
    if not label_lower:
        return []
    times = [
        frame["t"] for frame in entry["frames"]
        if any(c.lower() == label_lower for c in frame.get("objects", []))
    ]
    return _merge_adjacent_segments(times, frame_interval)
