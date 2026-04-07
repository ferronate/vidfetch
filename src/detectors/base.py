"""
Base detector interface for pluggable detection backends.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class Detection:
    """Represents a single detection result."""
    class_name: str
    confidence: float
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "t": round(self.timestamp, 2)
        }


class Detector(ABC):
    """
    Abstract base class for all detectors.
    Implement this interface to add new detection backends.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this detector."""
        pass
    
    @property
    @abstractmethod
    def supports_open_vocab(self) -> bool:
        """Whether this detector supports open-vocabulary detection."""
        pass
    
    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        prompt: Optional[str] = None,
        confidence_threshold: float = 0.25
    ) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame (RGB format)
            prompt: Text prompt for open-vocabulary detection (optional)
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def get_available_classes(self) -> List[str]:
        """Return list of available classes (for fixed-class detectors)."""
        pass
    
    def is_available(self) -> bool:
        """Check if this detector is available (dependencies installed)."""
        return True

    def detect_batch(
        self,
        frames: List[np.ndarray],
        prompt: Optional[str] = None,
        confidence_threshold: float = 0.25
    ) -> List[List[Detection]]:
        """Detect objects in a batch of frames.

        Default implementation calls `detect` for each frame. Backends that can
        efficiently perform batched inference (e.g., ONNX) should override this
        method to provide a faster implementation.
        """
        results: List[List[Detection]] = []
        for f in frames:
            results.append(self.detect(f, prompt=prompt, confidence_threshold=confidence_threshold))
        return results