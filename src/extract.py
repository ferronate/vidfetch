"""
Frame sampling and lightweight feature extraction (color histograms).
"""
from pathlib import Path
import cv2
import numpy as np


def sample_frames(video_path: str | Path, fps: float = 1.0, max_frames: int = 50) -> list[np.ndarray]:
    """
    Sample frames from a video at roughly the given fps, capped at max_frames.
    Returns list of BGR frames (numpy arrays).
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(video_fps / fps))
    frames = []
    idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


def sample_frames_with_time(
    video_path: str | Path, fps: float = 1.0, max_frames: int = 50
) -> list[tuple[np.ndarray, float]]:
    """
    Sample frames from a video with timestamps in seconds.
    Returns list of (frame, time_sec).
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(video_fps / fps))
    result = []
    idx = 0
    while len(result) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            t_sec = idx / video_fps
            result.append((frame, t_sec))
        idx += 1
    cap.release()
    return result


def frame_to_histogram(frame: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Convert a BGR frame to a compact HSV color histogram (1D vector).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_hist = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s_hist = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 256))[0]
    v_hist = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    hist /= hist.sum()
    return hist


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def video_to_feature(
    video_path: str | Path,
    fps: float = 1.0,
    max_frames: int = 50,
    bins: int = 16,
) -> np.ndarray:
    """
    Sample frames from a video (or load single image) and return one feature vector.
    """
    path = Path(video_path)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {path}")
        return frame_to_histogram(frame, bins=bins).astype(np.float32)
    frames = sample_frames(video_path, fps=fps, max_frames=max_frames)
    if not frames:
        return np.zeros(3 * bins, dtype=np.float32)
    hists = [frame_to_histogram(f, bins=bins) for f in frames]
    return np.mean(hists, axis=0).astype(np.float32)


def get_preset_color_feature(preset: str, bins: int = 16) -> np.ndarray:
    """
    Return a reference color histogram for filter presets.
    Presets: "warm" (red/orange), "cool" (blue), "bright" (high V), "dark" (low V).
    """
    n = 3 * bins
    out = np.zeros(n, dtype=np.float32)
    if preset == "warm":
        # Hue: red/orange (bins 0–3 in 0–180 range)
        out[0:4] = 1.0
        out[bins : bins + 4] = 0.8
        out[2 * bins : 2 * bins + bins] = 1.0
    elif preset == "cool":
        # Hue: blue (bins ~8–11)
        out[8:12] = 1.0
        out[bins + 8 : bins + 12] = 0.8
        out[2 * bins :] = 1.0
    elif preset == "bright":
        out[2 * bins :] = 1.0
        out[: 2 * bins] = 0.3
    elif preset == "dark":
        out[2 * bins :] = 0.2
        out[: 2 * bins] = 0.5
    else:
        out[:] = 1.0 / n
    out /= out.sum()
    return out
