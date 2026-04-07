"""
ONNX-based object detector for efficient CPU inference.
Uses quantized YOLO models with ONNX Runtime.
"""
from typing import List, Optional
import numpy as np
import logging
import cv2
from pathlib import Path

from .base import Detector, Detection
from src.utils import nms
from src.cpu_profile import get_cpu_profile

logger = logging.getLogger(__name__)

# COCO 80 class names — used for standard YOLOv8 ONNX models.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class ONNXDetector(Detector):
    """ONNX Runtime detector for efficient CPU object detection."""

    def __init__(
        self,
        model_path: str = "models/yolov8n.onnx",
        input_size: Optional[int] = None,
    ):
        profile = get_cpu_profile()
        self.model_path = Path(model_path)
        self.input_size = input_size if input_size is not None else profile.input_size
        self._session = None
        self._input_name = None
        self._output_names = None

    @property
    def name(self) -> str:
        return f"ONNX-YOLO ({self.model_path.stem})"

    @property
    def supports_open_vocab(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load the ONNX inference session."""
        if self._session is not None:
            return self._session

        import onnxruntime as ort

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        logger.info("Loading ONNX model: %s", self.model_path)

        profile = get_cpu_profile()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = profile.num_threads
        sess_options.inter_op_num_threads = max(1, profile.num_threads // 2)
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        logger.info("ONNX model loaded — input: %s, outputs: %s", self._input_name, self._output_names)
        return self._session

    # ------------------------------------------------------------------
    # Pre/post processing
    # ------------------------------------------------------------------

    def _preprocess_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Resize, normalise, and stack *frames* into an NCHW blob."""
        imgs = [
            cv2.resize(f, (self.input_size, self.input_size)).astype(np.float32) / 255.0
            for f in frames
        ]
        blob = np.stack(imgs, axis=0)          # N, H, W, C
        blob = np.transpose(blob, (0, 3, 1, 2))  # N, C, H, W
        return np.ascontiguousarray(blob)

    def _postprocess_single(
        self,
        predictions: np.ndarray,
        orig_h: int,
        orig_w: int,
        confidence_threshold: float,
    ) -> List[Detection]:
        """Decode one frame's raw predictions into Detection objects + NMS."""
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        detections: List[Detection] = []

        # YOLOv8 ONNX output shape: (84, 8400) where each COLUMN is a detection
        # Format per column: [x_center, y_center, width, height, class_0, class_1, ..., class_79]
        # Transpose to (8400, 84) for easier iteration
        if predictions.shape[0] == 84 and predictions.shape[1] > 84:
            predictions = predictions.T  # Now (8400, 84)

        for pred in predictions:
            # YOLOv8: No separate objectness score - class scores ARE the confidence
            class_scores = pred[4:]  # All class scores starting from index 4
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])  # Direct confidence, no multiplication

            if confidence < confidence_threshold:
                continue

            x_center, y_center, width, height = pred[0:4]
            x_center *= scale_x
            y_center *= scale_y
            width *= scale_x
            height *= scale_y

            x1 = max(0.0, x_center - width / 2)
            y1 = max(0.0, y_center - height / 2)
            x2 = min(float(orig_w), x_center + width / 2)
            y2 = min(float(orig_h), y_center + height / 2)

            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                )
            )

        return nms(detections, iou_threshold=0.45)

    # ------------------------------------------------------------------
    # Public detection API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        prompt: Optional[str] = None,
        confidence_threshold: float = 0.25,
    ) -> List[Detection]:
        """Detect objects in a single frame (delegates to detect_batch)."""
        results = self.detect_batch([frame], prompt=prompt, confidence_threshold=confidence_threshold)
        return results[0]

    def detect_batch(
        self,
        frames: List[np.ndarray],
        prompt: Optional[str] = None,
        confidence_threshold: float = 0.25,
    ) -> List[List[Detection]]:
        """Detect objects in a batch of frames via a single ONNX call."""
        session = self._load_model()

        try:
            original_shapes = [f.shape[:2] for f in frames]
            blob = self._preprocess_frames(frames)

            outputs = session.run(self._output_names, {self._input_name: blob})
            predictions = outputs[0]

            results: List[List[Detection]] = []

            if predictions.ndim == 3:
                for i in range(predictions.shape[0]):
                    orig_h, orig_w = original_shapes[i]
                    dets = self._postprocess_single(predictions[i], orig_h, orig_w, confidence_threshold)
                    results.append(dets)
            else:
                # Fallback for unexpected output shape
                orig_h, orig_w = original_shapes[0]
                results.append(self._postprocess_single(predictions, orig_h, orig_w, confidence_threshold))

            return results

        except Exception as e:
            logger.error("ONNX batch detection failed: %s", e)
            return [[] for _ in frames]

    def get_available_classes(self) -> List[str]:
        return list(COCO_CLASSES)

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
            return self.model_path.exists()
        except ImportError:
            return False