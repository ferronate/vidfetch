"""
Inference worker functions executed inside ProcessPoolExecutor workers.
This module lazily initializes per-process VideoDetector instances and exposes
functions that are safe to call from a ProcessPoolExecutor.

Functions are top-level so they can be pickled by multiprocessing.
"""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from src.detector import VideoDetector

logger = logging.getLogger(__name__)

# Per-worker cache of VideoDetector instances keyed by a tuple of config
_DETECTOR_CACHE: Dict[Tuple, VideoDetector] = {}


def _make_key(
    detector_type: str,
    confidence: float,
    sample_fps: float,
    enable_tracking: bool,
    enable_adaptive_sampling: bool,
    enable_temporal_aggregation: bool,
    batch_size: int
) -> Tuple:
    return (detector_type, float(confidence), float(sample_fps),
            bool(enable_tracking), bool(enable_adaptive_sampling), bool(enable_temporal_aggregation), int(batch_size))


def _ensure_detector(
    detector_type: str,
    confidence: float,
    sample_fps: float,
    enable_tracking: bool,
    enable_adaptive_sampling: bool,
    enable_temporal_aggregation: bool,
    batch_size: int = 1
) -> VideoDetector:
    """Get or create a VideoDetector for this worker process."""
    key = _make_key(detector_type, confidence, sample_fps,
                    enable_tracking, enable_adaptive_sampling, enable_temporal_aggregation, batch_size)
    if key not in _DETECTOR_CACHE:
        logger.info(f"[worker] Initializing VideoDetector: {key}")
        det = VideoDetector(
            detector_type=detector_type,
            confidence=confidence,
            sample_fps=sample_fps,
            enable_tracking=enable_tracking,
            enable_adaptive_sampling=enable_adaptive_sampling,
            enable_temporal_aggregation=enable_temporal_aggregation,
            batch_size=batch_size
        )
        _DETECTOR_CACHE[key] = det
    return _DETECTOR_CACHE[key]


def run_detection(
    video_path: str,
    detector_type: str = "auto",
    confidence: float = 0.10,
    sample_fps: float = 0.5,
    prompt: Optional[str] = None,
    enable_tracking: bool = True,
    enable_adaptive_sampling: bool = True,
    enable_temporal_aggregation: bool = True,
    batch_size: int = 1
) -> Dict:
    """Run detection on a single video and return a simple serializable dict.

    This function is intended to be called inside a ProcessPoolExecutor worker.
    It will lazily initialize a `VideoDetector` and reuse it for subsequent calls
    in the same worker process.
    """
    video_path = str(video_path)

    # Ensure detector exists in this worker
    detector = _ensure_detector(
        detector_type=detector_type,
        confidence=confidence,
        sample_fps=sample_fps,
        enable_tracking=enable_tracking,
        enable_adaptive_sampling=enable_adaptive_sampling,
        enable_temporal_aggregation=enable_temporal_aggregation,
        batch_size=batch_size,
    )

    # Run detection (returns unique_classes, timeline, pipeline_stats)
    unique_classes, timeline, pipeline_stats = detector.detect_video(video_path, prompt=prompt)

    # Build a JSON-serializable result dict
    result = {
        "video": str(video_path),
        "detector": detector.get_detector_info(),
        "pipeline_stats": pipeline_stats,
        "classes": unique_classes,
        "total_classes": len(unique_classes),
        "timeline": timeline,
        "total_detections": len(timeline),
        "prompt": prompt
    }

    return result


def index_videos_batch(
    video_paths: List[str],
    detector_type: str = "auto",
    confidence: float = 0.10,
    sample_fps: float = 0.5,
    prompt: Optional[str] = None,
    enable_tracking: bool = True,
    enable_adaptive_sampling: bool = True,
    enable_temporal_aggregation: bool = True
) -> Dict[str, Dict]:
    """Index a batch of videos and return a mapping video_id -> detection_result.

    This runs `run_detection` for each path sequentially inside the worker process.
    The worker-local detector will be reused across the batch for speed.
    """
    results = {}
    for p in video_paths:
        try:
            res = run_detection(
                p,
                detector_type=detector_type,
                confidence=confidence,
                sample_fps=sample_fps,
                prompt=prompt,
                enable_tracking=enable_tracking,
                enable_adaptive_sampling=enable_adaptive_sampling,
                enable_temporal_aggregation=enable_temporal_aggregation
            )
            video_id = Path(p).stem
            results[video_id] = res
        except Exception as e:
            # Keep error information per-video
            results[Path(p).stem] = {"error": str(e), "classes": [], "timeline": []}
    return results


def prewarm_detector(
    detector_type: str = "auto",
    confidence: float = 0.10,
    sample_fps: float = 0.5,
    enable_tracking: bool = True,
    enable_adaptive_sampling: bool = True,
    enable_temporal_aggregation: bool = True,
):
    """Public helper to pre-initialize a worker-local VideoDetector.

    This is intended to be submitted to the process pool to warm a worker
    (so that subsequent calls reuse a loaded model).
    Returns a small dict with detector info on success.
    """
    det = _ensure_detector(
        detector_type,
        confidence,
        sample_fps,
        enable_tracking,
        enable_adaptive_sampling,
        enable_temporal_aggregation,
    )

    # Force model/session load and perform a tiny synthetic inference if possible.
    try:
        # Call detector-specific load hook if present
        if hasattr(det, "_load_model"):
            try:
                det._load_model()
            except Exception as e:
                logger.warning(f"Detector _load_model() failed during prewarm: {e}")

        # Synthetic inference to trigger any lazy compilation (best-effort)
        try:
            import numpy as _np

            dummy = _np.zeros((480, 640, 3), dtype=_np.uint8)
            if hasattr(det, "detect"):
                try:
                    # Low-confidence, best-effort run; ignore results
                    det.detect(dummy, prompt=None, confidence_threshold=0.01)
                except Exception as e:
                    logger.debug(f"Synthetic detect() during prewarm failed: {e}")
        except Exception:
            # If numpy unavailable or synthetic run fails, ignore - model load is primary goal
            pass

        try:
            info = det.get_detector_info()
        except Exception:
            info = {"name": getattr(det, 'name', 'unknown')}
    except Exception as e:
        logger.exception(f"Prewarm failed: {e}")
        info = {"name": getattr(det, 'name', 'unknown')}

    return {"status": "warmed", "detector": info}
