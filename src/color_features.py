"""Video color feature extraction and similarity utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def _normalize_hist(hist: np.ndarray) -> np.ndarray:
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total <= 0:
        return hist
    return hist / total


def extract_color_summary(video_path: str | Path, samples: int = 20, max_side: int = 320) -> Optional[dict[str, Any]]:
    """Extract lightweight HSV-based color summary for a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return None

    sample_count = min(max(6, samples), frame_count)
    indices = np.linspace(0, max(0, frame_count - 1), num=sample_count, dtype=np.int32)

    hist_accum = np.zeros((36,), dtype=np.float64)
    warm_ratio_sum = 0.0
    cool_ratio_sum = 0.0
    bright_ratio_sum = 0.0
    dark_ratio_sum = 0.0
    sat_mean_sum = 0.0
    val_mean_sum = 0.0
    valid_frames = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        long_side = max(h, w)
        if long_side > max_side:
            scale = max_side / float(long_side)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        hist = cv2.calcHist([hue], [0], None, [36], [0, 180]).reshape(-1)
        hist_accum += hist.astype(np.float64)

        # OpenCV hue is [0,179]
        warm_mask = (hue <= 35) | (hue >= 150)
        cool_mask = (hue >= 45) & (hue <= 130)

        warm_ratio_sum += float(np.mean(warm_mask))
        cool_ratio_sum += float(np.mean(cool_mask))
        bright_ratio_sum += float(np.mean(val >= 170))
        dark_ratio_sum += float(np.mean(val <= 80))
        sat_mean_sum += float(np.mean(sat) / 255.0)
        val_mean_sum += float(np.mean(val) / 255.0)
        valid_frames += 1

    cap.release()

    if valid_frames == 0:
        return None

    hue_hist = _normalize_hist(hist_accum).tolist()

    return {
        "hue_hist": [float(v) for v in hue_hist],
        "warm_ratio": warm_ratio_sum / valid_frames,
        "cool_ratio": cool_ratio_sum / valid_frames,
        "bright_ratio": bright_ratio_sum / valid_frames,
        "dark_ratio": dark_ratio_sum / valid_frames,
        "saturation_mean": sat_mean_sum / valid_frames,
        "brightness_mean": val_mean_sum / valid_frames,
        "sampled_frames": valid_frames,
    }


def bhattacharyya_distance(hist_a: list[float], hist_b: list[float]) -> float:
    """Compute Bhattacharyya distance in [0,1] for normalized histograms."""
    a = np.array(hist_a, dtype=np.float64)
    b = np.array(hist_b, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 1.0

    # Renormalize defensively
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum <= 0 or b_sum <= 0:
        return 1.0
    a /= a_sum
    b /= b_sum

    coeff = float(np.sum(np.sqrt(a * b)))
    coeff = max(0.0, min(1.0, coeff))
    return float(np.sqrt(1.0 - coeff))


def passes_color_mode(summary: dict[str, Any], mode: str) -> bool:
    """Simple heuristic thresholds for color mode filters."""
    mode = (mode or "any").lower()
    if mode == "any":
        return True
    if mode == "warm":
        return float(summary.get("warm_ratio", 0.0)) >= 0.28
    if mode == "cool":
        return float(summary.get("cool_ratio", 0.0)) >= 0.28
    if mode == "bright":
        return float(summary.get("bright_ratio", 0.0)) >= 0.40
    if mode == "dark":
        return float(summary.get("dark_ratio", 0.0)) >= 0.35
    return True


def mode_distance(summary: dict[str, Any], mode: str) -> float:
    """Lower is better; used for ranking within a selected mode."""
    mode = (mode or "any").lower()
    if mode == "warm":
        return 1.0 - float(summary.get("warm_ratio", 0.0))
    if mode == "cool":
        return 1.0 - float(summary.get("cool_ratio", 0.0))
    if mode == "bright":
        return 1.0 - float(summary.get("bright_ratio", 0.0))
    if mode == "dark":
        return 1.0 - float(summary.get("dark_ratio", 0.0))
    return 0.1
