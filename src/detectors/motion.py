"""
Motion detection and scene change detection for video processing.
Detects motion regions and scene changes to optimize detection.
"""
from typing import List, Tuple, Optional
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class MotionDetector:
    """
    Detects motion in video frames.
    Used to focus detection on regions with activity.
    """
    
    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 16,
        detect_shadows: bool = False,
        min_motion_area: int = 200
    ):
        """
        Initialize motion detector.
        
        Args:
            history: Number of frames for background model
            var_threshold: Variance threshold for foreground detection
            detect_shadows: Whether to detect shadows
            min_motion_area: Minimum area (pixels) to consider as motion
        """
        self.min_motion_area = min_motion_area
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
    
    def detect_motion_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect regions with motion in the frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of bounding boxes (x, y, w, h) for motion regions
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Threshold to remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter by area and get bounding boxes
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_motion_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x, y, w, h))
        
        return motion_regions
    
    def get_motion_score(self, frame: np.ndarray) -> float:
        """
        Calculate overall motion score for the frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Motion score (0.0 to 1.0)
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Count foreground pixels
        fg_pixels = np.sum(fg_mask > 127)
        total_pixels = frame.shape[0] * frame.shape[1]
        
        return fg_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def should_sample_frame(self, frame: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Determine if frame should be sampled based on motion.
        
        Args:
            frame: Input frame
            threshold: Motion threshold for sampling
            
        Returns:
            True if frame should be sampled
        """
        motion_score = self.get_motion_score(frame)
        return motion_score >= threshold


class SceneChangeDetector:
    """
    Detects scene changes (cuts, transitions) in video.
    Resets tracking when scene changes.
    """
    
    def __init__(
        self,
        histogram_bins: int = 64,
        threshold: float = 0.35,
        min_scene_length: int = 6
    ):
        """
        Initialize scene change detector.
        
        Args:
            histogram_bins: Number of bins for histogram comparison
            threshold: Threshold for scene change detection
            min_scene_length: Minimum frames between scene changes
        """
        self.histogram_bins = histogram_bins
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        
        self.prev_histogram = None
        self.frames_since_change = 0
    
    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate color histogram for frame."""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [self.histogram_bins, self.histogram_bins, self.histogram_bins],
            [0, 180, 0, 256, 0, 256]
        )
        
        # Normalize
        cv2.normalize(hist, hist)
        
        return hist
    
    def detect_scene_change(self, frame: np.ndarray) -> bool:
        """
        Detect if scene has changed from previous frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            True if scene change detected
        """
        self.frames_since_change += 1
        
        # Calculate current histogram
        current_histogram = self._calculate_histogram(frame)
        
        # First frame
        if self.prev_histogram is None:
            self.prev_histogram = current_histogram
            return False
        
        # Compare histograms
        correlation = cv2.compareHist(
            self.prev_histogram,
            current_histogram,
            cv2.HISTCMP_CORREL
        )
        
        # Scene change if correlation is low
        is_scene_change = (1 - correlation) > self.threshold
        
        # Check minimum scene length
        if is_scene_change and self.frames_since_change >= self.min_scene_length:
            self.prev_histogram = current_histogram
            self.frames_since_change = 0
            logger.info(f"Scene change detected (correlation: {correlation:.3f})")
            return True
        
        self.prev_histogram = current_histogram
        return False
    
    def reset(self):
        """Reset scene change detector."""
        self.prev_histogram = None
        self.frames_since_change = 0


class AdaptiveSampler:
    """
    Adaptive frame sampler based on motion and scene changes.
    Samples more frames during activity, fewer during static scenes.
    """
    
    def __init__(
        self,
        base_fps: Optional[float] = 1.0,
        max_fps: Optional[float] = 5.0,
        motion_threshold: float = 0.005,
        scene_change_boost: int = 3
    ):
        """
        Initialize adaptive sampler.
        
        Args:
            base_fps: Base sampling rate (frames per second)
            max_fps: Maximum sampling rate during high motion
            motion_threshold: Motion threshold for increased sampling
            scene_change_boost: Extra frames to sample after scene change
        """
        self.base_fps = float(base_fps) if base_fps and base_fps > 0 else 0.5
        self.max_fps = float(max_fps) if max_fps and max_fps > 0 else 2.0
        if self.max_fps < self.base_fps:
            self.max_fps = self.base_fps
        self.motion_threshold = motion_threshold
        self.scene_change_boost = scene_change_boost
        
        self.motion_detector = MotionDetector()
        self.scene_detector = SceneChangeDetector()
        
        self.frames_since_sample = 0
        self.consecutive_static = 0
    
    def should_sample(
        self,
        frame: np.ndarray,
        frame_number: int,
        fps: float
    ) -> bool:
        """
        Determine if current frame should be sampled.
        
        Args:
            frame: Current frame
            frame_number: Current frame number
            fps: Video FPS
            
        Returns:
            True if frame should be sampled
        """
        self.frames_since_sample += 1
        
        # Normalize video FPS defensively to avoid invalid interval math.
        effective_fps = float(fps) if fps and fps > 0 else 30.0
        high_motion_fps = max(self.max_fps, 0.01)
        base_sampling_fps = max(self.base_fps, 0.01)

        # Check for scene change
        scene_changed = self.scene_detector.detect_scene_change(frame)
        
        # Check for motion
        has_motion = self.motion_detector.should_sample_frame(
            frame,
            self.motion_threshold
        )
        
        # Calculate sampling interval
        if scene_changed:
            # Sample immediately after scene change
            interval = 1
            self.consecutive_static = 0
        elif has_motion:
            # High motion - sample more frequently
            interval = max(1, int(effective_fps / high_motion_fps))
            self.consecutive_static = 0
        else:
            # Static scene - sample less frequently
            self.consecutive_static += 1
            interval = max(1, int(effective_fps / base_sampling_fps))
            # Increase interval for very static scenes
            if self.consecutive_static > 30:
                interval = min(interval * 2, max(1, int(effective_fps / 0.25)))
        
        # Should we sample this frame?
        should_sample = self.frames_since_sample >= interval
        
        if should_sample:
            self.frames_since_sample = 0
        
        return should_sample
    
    def reset(self):
        """Reset adaptive sampler."""
        self.motion_detector = MotionDetector()
        self.scene_detector.reset()
        self.frames_since_sample = 0
        self.consecutive_static = 0