"""
Detector manager — selects the right detector based on CPU capabilities.

Uses ``cpu_profile.get_cpu_profile()`` to decide which model and settings
are appropriate for the host hardware.  Callers can also request a specific
detector type explicitly.
"""
from typing import Optional, Dict, Any
import logging

from .base import Detector
from .onnx_detector import ONNXDetector
from .yolo import YOLOWorldDetector, YOLODetector
from src.cpu_profile import get_cpu_profile, MODEL_REGISTRY

logger = logging.getLogger(__name__)


class DetectorManager:
    """Manages detector selection based on CPU capabilities and user preferences."""

    def __init__(self, detector_type: str = "auto"):
        """
        Args:
            detector_type: 'auto', 'onnx', 'yolo-world', 'yolo', or 'legacy'.
        """
        self.detector_type = detector_type
        self._detector: Optional[Detector] = None
        self._profile = get_cpu_profile()

    # ------------------------------------------------------------------
    # Internal: build the right Detector instance
    # ------------------------------------------------------------------

    def _build_for_profile(self) -> Detector:
        """Build a detector based on the current CPU profile."""
        profile = self._profile
        entry = MODEL_REGISTRY[profile.model_key]
        model_type = entry["type"]

        if model_type == "onnx":
            logger.info(
                "Auto-selected ONNX detector (%s) for %s-tier CPU",
                profile.model_key,
                profile.tier,
            )
            return ONNXDetector(
                model_path=str(profile.model_path),
                input_size=profile.input_size,
            )

        if model_type == "yolo-world":
            logger.info(
                "Auto-selected YOLO-World detector for %s-tier CPU",
                profile.tier,
            )
            return YOLOWorldDetector(
                model_size="s",
                model_path=str(profile.model_path),
            )

        # Default: standard YOLO (object365, coco-nano, etc.)
        logger.info(
            "Auto-selected YOLO detector (%s) for %s-tier CPU",
            profile.model_key,
            profile.tier,
        )
        return YOLODetector(model_path=str(profile.model_path))

    def _select_detector(self) -> Detector:
        """Select the best detector based on type preference."""
        dt = self.detector_type.lower()

        if dt == "auto":
            return self._build_for_profile()

        if dt == "onnx":
            profile = self._profile
            return ONNXDetector(
                model_path=str(profile.model_path)
                if MODEL_REGISTRY.get(profile.model_key, {}).get("type") == "onnx"
                else str(self._profile.model_path.parent / "yolov8n.onnx"),
                input_size=profile.input_size,
            )

        if dt == "yolo-world":
            return YOLOWorldDetector(
                model_size="s",
                model_path=str(self._profile.model_path) if MODEL_REGISTRY.get(self._profile.model_key, {}).get("type") == "yolo-world" else None,
            )

        if dt in ("yolo", "legacy"):
            return YOLODetector(model_path=str(self._profile.model_path))

        logger.warning("Unknown detector type '%s', falling back to auto", dt)
        self.detector_type = "auto"
        return self._build_for_profile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_detector(self) -> Detector:
        """Return the selected Detector (lazy-initialised)."""
        if self._detector is None:
            self._detector = self._select_detector()

            if not self._detector.is_available():
                logger.warning(
                    "%s not available, falling back to YOLOv8-nano",
                    self._detector.name,
                )
                self._detector = YOLODetector(model_size="n")

        return self._detector

    def get_detector_info(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict describing the active detector."""
        detector = self.get_detector()
        profile = self._profile
        return {
            "name": detector.name,
            "type": self.detector_type,
            "cpu_tier": profile.tier,
            "cpu_brand": profile.cpu_brand,
            "cpu_cores": profile.cores,
            "model_key": profile.model_key,
            "input_size": profile.input_size,
            "batch_size": profile.batch_size,
            "num_threads": profile.num_threads,
            "supports_open_vocab": detector.supports_open_vocab,
            "available": detector.is_available(),
        }

    def set_detector(self, detector_type: str):
        """Change the detector type (forces re-selection on next get)."""
        self.detector_type = detector_type
        self._detector = None