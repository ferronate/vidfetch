"""
Pluggable detector architecture for vidfetch.
Supports multiple detection backends with automatic CPU-driven selection.
Includes object tracking, temporal aggregation, and adaptive sampling.
"""

from .base import Detector, Detection
from .onnx_detector import ONNXDetector
from .yolo import YOLOWorldDetector, YOLODetector
from .manager import DetectorManager
from .tracking import SORTTracker, TemporalAggregator
from .motion import MotionDetector, SceneChangeDetector, AdaptiveSampler

__all__ = [
    "Detector",
    "Detection",
    "ONNXDetector",
    "YOLOWorldDetector",
    "YOLODetector",
    "DetectorManager",
    "SORTTracker",
    "TemporalAggregator",
    "MotionDetector",
    "SceneChangeDetector",
    "AdaptiveSampler",
]
