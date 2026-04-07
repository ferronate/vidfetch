"""
Consolidated YOLO detector implementations.

Provides two detector classes:
- `YOLOWorldDetector` (open-vocab)
- `YOLODetector` (legacy/fixed-classes)

This module centralizes YOLO-based detector implementations used
by the application. Historical split files were removed; callers
should import detectors from `src.detectors`.
"""
from typing import List, Optional
import logging
import numpy as np
import os

from src.config import CPU_ONLY, FORCE_YOLO_CPU

from .base import Detector, Detection

logger = logging.getLogger(__name__)


class YOLOWorldDetector(Detector):
    """
    YOLO-World detector for open-vocabulary object detection.
    """

    def __init__(self, model_size: str = "s", model_path: Optional[str] = None):
        self.model_size = model_size
        self.model_path = model_path
        self._model = None
        self._classes: List[str] = []

    @property
    def name(self) -> str:
        return f"YOLO-World ({self.model_size})"

    @property
    def supports_open_vocab(self) -> bool:
        return True

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                from src.cpu_profile import get_cpu_profile

                # Use provided local path or fall back to filename
                model_source = self.model_path if self.model_path else f"yolov8{self.model_size}-world.pt"
                logger.info(f"Loading YOLO-World model: {model_source}")
                
                # Set CPU thread environment variables before loading
                profile = get_cpu_profile()
                if CPU_ONLY or FORCE_YOLO_CPU:
                    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                    os.environ.setdefault("OMP_NUM_THREADS", str(profile.num_threads))
                    os.environ.setdefault("MKL_NUM_THREADS", str(profile.num_threads))
                    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(profile.num_threads))
                    self._model = YOLO(model_source)
                    # Optimize YOLO for CPU inference (removed 'threads' - not valid in newer ultralytics)
                    self._model.overrides.update({
                        'half': False,  # Disable FP16 on CPU
                    })
                else:
                    self._model = YOLO(model_source)
                logger.info("YOLO-World model loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to load YOLO-World: {e}")
                raise
            except Exception as e:
                logger.error(f"Error loading YOLO-World model: {e}")
                raise

        return self._model

    def detect(self, frame: np.ndarray, prompt: Optional[str] = None, confidence_threshold: float = 0.25) -> List[Detection]:
        model = self._load_model()

        try:
            if prompt:
                classes = [c.strip() for c in prompt.split(",") if c.strip()]
                self.set_classes(classes)

            results = model(frame, conf=confidence_threshold, verbose=False)

            detections: List[Detection] = []
            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name = model.names.get(class_id, str(class_id))

                    bbox = None
                    if box.xyxy is not None:
                        # xyxy is a tensor; convert to list
                        try:
                            bbox = box.xyxy[0].tolist()
                        except Exception:
                            bbox = None

                    detections.append(Detection(class_name=class_name, confidence=confidence, bbox=bbox))

            return detections
        except Exception as e:
            logger.error(f"YOLO-World detection failed: {e}")
            return []

    def get_available_classes(self) -> List[str]:
        return self._classes

    def set_classes(self, classes: List[str]):
        self._classes = classes
        if self._model is not None:
            try:
                self._model.set_classes(classes)
            except Exception:
                # Some ultralytics versions may not support set_classes for world models
                pass

    def is_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
            return True
        except Exception:
            return False


class YOLODetector(Detector):
    """
    Legacy YOLO detector (fixed-class YOLOv8 or custom model path).
    """

    def __init__(self, model_size: str = "n", model_path: Optional[str] = None):
        self.model_size = model_size
        self.model_path = model_path
        self._model = None

    @property
    def name(self) -> str:
        if self.model_path:
            return "YOLO Custom"
        return f"YOLOv8 ({self.model_size})"

    @property
    def supports_open_vocab(self) -> bool:
        return False

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                from src.cpu_profile import get_cpu_profile

                model_name = self.model_path if self.model_path else f"yolov8{self.model_size}.pt"
                logger.info(f"Loading YOLO model: {model_name}")
                
                # Set CPU thread environment variables before loading
                profile = get_cpu_profile()
                if CPU_ONLY or FORCE_YOLO_CPU:
                    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                    os.environ.setdefault("OMP_NUM_THREADS", str(profile.num_threads))
                    os.environ.setdefault("MKL_NUM_THREADS", str(profile.num_threads))
                    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(profile.num_threads))
                    self._model = YOLO(model_name)
                    # Optimize YOLO for CPU inference (removed 'threads' - not valid in newer ultralytics)
                    self._model.overrides.update({
                        'half': False,  # Disable FP16 on CPU
                    })
                else:
                    self._model = YOLO(model_name)
                logger.info(f"YOLO model loaded with {len(self._model.names)} classes")
            except ImportError as e:
                logger.error(f"Failed to load YOLO: {e}")
                raise
            except Exception as e:
                logger.error(f"Error loading YOLO model: {e}")
                raise

        return self._model

    def detect(self, frame: np.ndarray, prompt: Optional[str] = None, confidence_threshold: float = 0.25) -> List[Detection]:
        model = self._load_model()

        try:
            results = model(frame, conf=confidence_threshold, verbose=False)

            detections: List[Detection] = []
            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name = model.names.get(class_id, str(class_id))

                    bbox = None
                    if box.xyxy is not None:
                        try:
                            bbox = box.xyxy[0].tolist()
                        except Exception:
                            bbox = None

                    detections.append(Detection(class_name=class_name, confidence=confidence, bbox=bbox))

            return detections
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def get_available_classes(self) -> List[str]:
        model = self._load_model()
        try:
            return list(model.names.values())
        except Exception:
            return []

    def is_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
            return True
        except Exception:
            return False
