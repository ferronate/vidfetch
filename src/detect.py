"""
Lightweight object detection using OpenCV DNN + MobileNet-SSD (Pascal VOC, 21 classes).
Runs on CPU; no GPU required. Model files go in models/ (see scripts/download_detector_model.py).
"""
from pathlib import Path
import cv2
import numpy as np

# Pascal VOC class names (index 0 = background)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv monitor",
]

# Input size for MobileNet-SSD
INPUT_WIDTH = 300
INPUT_HEIGHT = 300
CONFIDENCE_THRESHOLD = 0.6  # Raise to reduce false positives (e.g. dog on shoes)


def _get_model_paths(models_dir: Path) -> tuple[Path, Path]:
    models_dir = Path(models_dir)
    prototxt = models_dir / "MobileNetSSD_deploy.prototxt"
    caffemodel = models_dir / "MobileNetSSD_deploy.caffemodel"
    return prototxt, caffemodel


def load_net(models_dir: str | Path):
    """Load MobileNet-SSD Caffe model. Raises FileNotFoundError if files missing."""
    prototxt, caffemodel = _get_model_paths(models_dir)
    if not prototxt.exists():
        raise FileNotFoundError(
            f"Prototxt not found: {prototxt}. Run: python -m scripts.download_detector_model"
        )
    if not caffemodel.exists():
        raise FileNotFoundError(
            f"Caffemodel not found: {caffemodel}. Run: python -m scripts.download_detector_model"
        )
    return cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))


def detect_objects(
    frame: np.ndarray,
    net: cv2.dnn.Net,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[tuple[str, float]]:
    """
    Run object detection on a BGR frame. Returns list of (class_name, confidence).
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 0.007843, (INPUT_WIDTH, INPUT_HEIGHT), (127.5, 127.5, 127.5), swapRB=False
    )
    net.setInput(blob)
    detections = net.forward()
    results = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < confidence_threshold:
            continue
        class_id = int(detections[0, 0, i, 1])
        if 0 <= class_id < len(VOC_CLASSES):
            results.append((VOC_CLASSES[class_id], conf))
    return results


def video_to_object_set(
    video_path: str | Path,
    net: cv2.dnn.Net,
    fps: float = 0.5,
    max_frames: int = 30,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> set[str]:
    """
    Sample frames from video, run detection, return set of detected class names (excluding background).
    """
    from .extract import sample_frames

    frames = sample_frames(video_path, fps=fps, max_frames=max_frames)
    classes = set()
    for frame in frames:
        for name, _ in detect_objects(frame, net, confidence_threshold):
            if name != "background":
                classes.add(name)
    return classes


def video_to_object_timeline(
    video_path: str | Path,
    net: cv2.dnn.Net,
    fps: float = 0.5,
    max_frames: int = 30,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[set[str], list[dict]]:
    """
    Sample frames with timestamps, run detection. Returns (set of all classes, list of {t, objects}).
    Each entry has "t" in seconds and "objects" as list of class names (excluding background).
    """
    from .extract import sample_frames_with_time

    frames_with_time = sample_frames_with_time(video_path, fps=fps, max_frames=max_frames)
    classes = set()
    timeline = []
    for frame, t_sec in frames_with_time:
        names = [
            name for name, _ in detect_objects(frame, net, confidence_threshold)
            if name != "background"
        ]
        for name in names:
            classes.add(name)
        timeline.append({"t": round(t_sec, 2), "objects": names})
    return classes, timeline
