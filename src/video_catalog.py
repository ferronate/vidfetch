"""
Simple cached video catalog to avoid repeated filesystem scans.
Provides a TTL-based cache and helper to find videos by id.
"""
import time
from pathlib import Path
from typing import List, Dict, Optional

from .utils import DATA_DIR, VIDEO_EXTENSIONS

_CACHE: Dict[str, object] = {"videos": [], "ts": 0.0}


def _scan_videos() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    return [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]


def get_video_files_cached(ttl: float = 5.0) -> List[Path]:
    """Return list of video files, cached for ``ttl`` seconds."""
    now = time.time()
    if now - _CACHE["ts"] > ttl:
        _CACHE["videos"] = _scan_videos()
        _CACHE["ts"] = now
    return _CACHE["videos"]


def find_video_by_id(video_id: str) -> Optional[Path]:
    """Return a Path for the given video_id (stem), or None if not found."""
    for p in get_video_files_cached():
        if p.stem == video_id:
            return p
    return None
