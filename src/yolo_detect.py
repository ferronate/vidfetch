"""
Object detection using YOLOv8 (Ultralytics), COCO-trained. Real detections with confidence,
not image-text similarity. Runs on CPU. Install: pip install ultralytics
"""
from pathlib import Path
import numpy as np

# Confidence 0.5: good recall for person/objects; false positives filtered by blacklist + min_frames
DEFAULT_CONFIDENCE = 0.5

# COCO classes YOLO often gets wrong in video (e.g. skateboard→toilet, lab→cake, flatbread→sandwich). Never tag these.
BLACKLIST = {"toilet", "sports ball", "sandwich", "cake", "cell phone", "remote", "keyboard", "mouse", "toaster", "hair drier", "toothbrush"}


def _load_model(model_size: str = "s"):
    """Load YOLOv8 (COCO 80 classes). 's' = small (better accuracy), 'n' = nano (faster)."""
    from ultralytics import YOLO
    name = f"yolov8{model_size}.pt"
    return YOLO(name)


def _detect_frame(
    frame: np.ndarray,
    model,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[str]:
    """Run YOLO on one BGR frame. Return list of detected class names (above confidence)."""
    results = model(frame, conf=confidence, verbose=False)
    names = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            n = getattr(model, "names", None)
            if isinstance(n, dict):
                name = n.get(cls_id, str(cls_id))
            elif isinstance(n, (list, tuple)) and cls_id < len(n):
                name = n[cls_id]
            else:
                name = str(cls_id)
            names.append(name)
    return list(dict.fromkeys(names))  # unique, order preserved


def video_to_object_timeline(
    video_path: str | Path,
    fps: float = 0.5,
    max_frames: int = 30,
    confidence: float = DEFAULT_CONFIDENCE,
    min_frames_per_class: int = 2,
    model_size: str = "s",
) -> tuple[set[str], list[dict]]:
    """
    Sample frames with timestamps, run YOLOv8 detection. Returns (set of all classes, list of {t, objects}).
    model_size: 's' (small, better accuracy) or 'n' (nano, faster). Blacklisted classes (toilet, sports ball, etc.) are never added.
    """
    from .extract import sample_frames_with_time

    model = _load_model(model_size)
    frames_with_time = sample_frames_with_time(video_path, fps=fps, max_frames=max_frames)
    class_frame_count = {}
    timeline = []
    for frame, t_sec in frames_with_time:
        names = _detect_frame(frame, model, confidence=confidence)
        for name in names:
            if name in BLACKLIST:
                continue
            class_frame_count[name] = class_frame_count.get(name, 0) + 1
        timeline.append({"t": round(t_sec, 2), "objects": [n for n in names if n not in BLACKLIST]})
    # Only keep classes seen in at least min_frames_per_class frames
    classes = {c for c, n in class_frame_count.items() if n >= min_frames_per_class}
    for entry in timeline:
        entry["objects"] = [o for o in entry["objects"] if o in classes]
    return classes, timeline
