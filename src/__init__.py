"""Lightweight video object detection using YOLO."""

from .detector import VideoDetector, detect_objects

__all__ = ["VideoDetector", "detect_objects"]