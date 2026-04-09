"""
Shared utilities for vidfetch.

Single-source-of-truth definitions that are used across multiple modules.
Import from here instead of duplicating constants or helper functions.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Repo root (one level above src/)
ROOT = Path(__file__).resolve().parent.parent

def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    return Path(value).expanduser().resolve()


DATA_DIR = _env_path("VF_DATA_DIR", ROOT / "data")
INDEX_DIR = _env_path("VF_INDEX_DIR", ROOT / "index_store")
MODELS_DIR = _env_path("VF_MODELS_DIR", ROOT / "models")
INDEX_FILE = INDEX_DIR / "detection_results.json"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection-over-Union between two [x1, y1, x2, y2] boxes.

    Used by tracking, NMS, and anywhere else that needs box overlap.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def nms(detections, iou_threshold: float = 0.45):
    """Apply class-aware Non-Max Suppression.

    Args:
        detections: List of objects with .confidence, .bbox, .class_name attrs
                    (e.g. Detection dataclass instances).
        iou_threshold: IoU above which the lower-confidence box is suppressed.

    Returns:
        Filtered list of detections.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

    keep = []
    while sorted_dets:
        current = sorted_dets.pop(0)
        keep.append(current)

        remaining = []
        for det in sorted_dets:
            # Only suppress same-class overlaps
            if det.class_name == current.class_name:
                iou = calculate_iou(current.bbox, det.bbox)
                if iou >= iou_threshold:
                    continue  # suppress
            remaining.append(det)

        sorted_dets = remaining

    return keep
