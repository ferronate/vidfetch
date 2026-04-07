"""
Minimal video object detector using pluggable detector architecture.
Supports YOLO-World (open-vocabulary) and YOLO (fixed classes) with automatic CPU selection.
Includes object tracking, temporal aggregation, adaptive sampling, and motion detection.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
import cv2
import numpy as np

from .detectors import DetectorManager, Detection
from .detectors.tracking import SORTTracker, TemporalAggregator
from .detectors.motion import AdaptiveSampler
from .cpu_profile import get_cpu_profile

logger = logging.getLogger(__name__)


class VideoDetector:
    """Video object detector using pluggable detector architecture with tracking."""
    
    def __init__(
        self,
        detector_type: str = "auto",
        confidence: float = 0.15,
        sample_fps: Optional[float] = None,
        enable_tracking: bool = True,
        enable_adaptive_sampling: bool = True,
        enable_temporal_aggregation: bool = True,
        batch_size: int = 1
    ):
        """
        Initialize video detector.
        
        Args:
            detector_type: Detector type - 'auto', 'yolo-world', 'yolo', 'legacy'
            confidence: Detection confidence threshold
            sample_fps: Base frames per second to sample
            enable_tracking: Enable object tracking across frames
            enable_adaptive_sampling: Enable motion-based adaptive sampling
            enable_temporal_aggregation: Enable voting across frames
        """
        self.detector_type = detector_type
        self.confidence = confidence
        # Preserve auto behavior when sample_fps is omitted, but never allow invalid values.
        resolved_sample_fps = sample_fps
        if resolved_sample_fps is None or resolved_sample_fps <= 0:
            profile_fps = get_cpu_profile().sample_fps
            if profile_fps and profile_fps > 0:
                resolved_sample_fps = profile_fps
                logger.info(
                    "Sampling FPS not provided or invalid (%s); using CPU profile default %.2f",
                    sample_fps,
                    resolved_sample_fps,
                )
            else:
                resolved_sample_fps = 0.5
                logger.warning(
                    "Sampling FPS not provided or invalid (%s) and CPU profile default unavailable; using fallback %.2f",
                    sample_fps,
                    resolved_sample_fps,
                )

        self.sample_fps = float(resolved_sample_fps)
        self.enable_tracking = enable_tracking
        self.enable_adaptive_sampling = enable_adaptive_sampling
        self.enable_temporal_aggregation = enable_temporal_aggregation
        self.batch_size = int(batch_size)
        
        # Initialize detector manager
        self.manager = DetectorManager(detector_type=detector_type)
        self.detector = self.manager.get_detector()
        
        # Initialize tracking components
        self.tracker = SORTTracker() if enable_tracking else None
        self.aggregator = TemporalAggregator() if enable_temporal_aggregation else None
        self.sampler = AdaptiveSampler(base_fps=self.sample_fps) if enable_adaptive_sampling else None
        
        logger.info(f"Initialized {self.detector.name} detector")
        logger.info(f"  Tracking: {enable_tracking}")
        logger.info(f"  Adaptive sampling: {enable_adaptive_sampling}")
        logger.info(f"  Temporal aggregation: {enable_temporal_aggregation}")
        logger.info(f"  Batch size: {self.batch_size}")
    
    def detect_video(
        self,
        video_path: str | Path,
        prompt: Optional[str] = None
    ) -> Tuple[List[str], List[Dict], Dict[str, int]]:
        """
        Detect objects in a video with tracking and temporal aggregation.
        
        Args:
            video_path: Path to video file
            prompt: Text prompt for detection (e.g., "person, car, dog")
            
        Returns:
            Tuple of (unique_classes, detections_timeline)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle videos that don't report FPS properly
        if not fps or fps <= 0:
            fps = 30.0  # Default fallback
            logger.warning(f"Video {video_path.name} reports invalid FPS ({fps}), defaulting to {fps}")
        
        all_classes = set()
        timeline = []
        frame_count = 0
        sampled_frames = 0
        raw_detection_frames = 0
        raw_detection_objects = 0
        tracked_detection_frames = 0
        tracked_detection_objects = 0
        aggregated_detection_frames = 0
        aggregated_detection_objects = 0
        
        logger.info(f"Processing {video_path.name}: {fps:.1f} FPS, {total_frames} frames")
        
        # Reset components
        if self.tracker:
            self.tracker.reset()
        if self.aggregator:
            self.aggregator.reset()
        if self.sampler:
            self.sampler.reset()

        # Buffers for batched processing
        frames_buffer: List[np.ndarray] = []
        frame_numbers_buffer: List[int] = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Determine if we should sample this frame
            should_sample = True
            if self.sampler:
                should_sample = self.sampler.should_sample(frame, frame_count, fps)
            
            if should_sample:
                sampled_frames += 1
                # Convert BGR to RGB and buffer for batched detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append(frame_rgb)
                frame_numbers_buffer.append(frame_count)

                # If buffer full, run batched detection
                if len(frames_buffer) >= max(1, self.batch_size):
                    try:
                        if hasattr(self.detector, "detect_batch"):
                            batch_results = self.detector.detect_batch(
                                frames_buffer,
                                prompt=prompt,
                                confidence_threshold=self.confidence
                            )
                        else:
                            batch_results = [
                                self.detector.detect(f, prompt=prompt, confidence_threshold=self.confidence)
                                for f in frames_buffer
                            ]
                    except Exception as e:
                        logger.exception(f"Batched detection failed: {e}")
                        batch_results = [[] for _ in frames_buffer]

                    # Process each frame's detections in order
                    for dets, fnum in zip(batch_results, frame_numbers_buffer):
                        if dets:
                            raw_detection_frames += 1
                            raw_detection_objects += len(dets)
                        # Debug logging
                        if fnum < 5 or (fnum % 30 == 0):
                            logger.info(f"Frame {fnum}: {len(dets)} detections (confidence={self.confidence})")
                            for det in dets[:3]:
                                logger.info(f"  - {det.class_name}: {det.confidence:.3f}")

                        # Convert to dict format for tracking
                        det_dicts = []
                        for det in dets:
                            det_dict = {
                                "class": det.class_name,
                                "confidence": det.confidence,
                                "bbox": det.bbox if det.bbox else [0, 0, 100, 100]
                            }
                            det_dicts.append(det_dict)

                        # Apply tracking
                        if self.tracker:
                            tracked_dets = self.tracker.update(det_dicts)
                        else:
                            tracked_dets = det_dicts

                        if tracked_dets:
                            tracked_detection_frames += 1
                            tracked_detection_objects += len(tracked_dets)

                        # Add to temporal aggregator
                        timestamp = fnum / fps if fps else fnum / 30.0
                        if self.aggregator:
                            self.aggregator.add_frame(tracked_dets, timestamp)
                            # Get aggregated detections
                            aggregated_dets = self.aggregator.get_aggregated_detections()
                        else:
                            aggregated_dets = tracked_dets

                        if aggregated_dets:
                            aggregated_detection_frames += 1
                            aggregated_detection_objects += len(aggregated_dets)

                        # Extract classes
                        frame_classes = [d["class"] for d in aggregated_dets]
                        all_classes.update(frame_classes)

                        # Add to timeline
                        if aggregated_dets:
                            timeline.append({
                                "t": round(timestamp, 2),
                                "objects": aggregated_dets
                            })

                    # Clear buffers
                    frames_buffer = []
                    frame_numbers_buffer = []
            
            frame_count += 1

        # After reading all frames, process any remaining buffered frames once
        if frames_buffer:
            try:
                if hasattr(self.detector, "detect_batch"):
                    batch_results = self.detector.detect_batch(
                        frames_buffer,
                        prompt=prompt,
                        confidence_threshold=self.confidence
                    )
                else:
                    batch_results = [
                        self.detector.detect(f, prompt=prompt, confidence_threshold=self.confidence)
                        for f in frames_buffer
                    ]
            except Exception as e:
                logger.exception(f"Batched detection (final) failed: {e}")
                batch_results = [[] for _ in frames_buffer]

            for dets, fnum in zip(batch_results, frame_numbers_buffer):
                if dets:
                    raw_detection_frames += 1
                    raw_detection_objects += len(dets)
                if fnum < 5 or (fnum % 30 == 0):
                    logger.info(f"Frame {fnum}: {len(dets)} detections (confidence={self.confidence})")
                    for det in dets[:3]:
                        logger.info(f"  - {det.class_name}: {det.confidence:.3f}")

                det_dicts = []
                for det in dets:
                    det_dict = {
                        "class": det.class_name,
                        "confidence": det.confidence,
                        "bbox": det.bbox if det.bbox else [0, 0, 100, 100]
                    }
                    det_dicts.append(det_dict)

                if self.tracker:
                    tracked_dets = self.tracker.update(det_dicts)
                else:
                    tracked_dets = det_dicts

                if tracked_dets:
                    tracked_detection_frames += 1
                    tracked_detection_objects += len(tracked_dets)

                timestamp = fnum / fps if fps else fnum / 30.0
                if self.aggregator:
                    self.aggregator.add_frame(tracked_dets, timestamp)
                    aggregated_dets = self.aggregator.get_aggregated_detections()
                else:
                    aggregated_dets = tracked_dets

                if aggregated_dets:
                    aggregated_detection_frames += 1
                    aggregated_detection_objects += len(aggregated_dets)

                frame_classes = [d["class"] for d in aggregated_dets]
                all_classes.update(frame_classes)

                if aggregated_dets:
                    timeline.append({
                        "t": round(timestamp, 2),
                        "objects": aggregated_dets
                    })

        cap.release()
        
        # Reset components
        if self.tracker:
            self.tracker.reset()
        if self.aggregator:
            self.aggregator.reset()
        
        unique_classes = sorted(list(all_classes))
        logger.info(f"Detected {len(unique_classes)} unique objects: {', '.join(unique_classes[:10])}{'...' if len(unique_classes) > 10 else ''}")
        pipeline_stats = {
            "video_total_frames": total_frames,
            "sampled_frames": sampled_frames,
            "raw_detection_frames": raw_detection_frames,
            "raw_detection_objects": raw_detection_objects,
            "tracked_detection_frames": tracked_detection_frames,
            "tracked_detection_objects": tracked_detection_objects,
            "aggregated_detection_frames": aggregated_detection_frames,
            "aggregated_detection_objects": aggregated_detection_objects,
            "timeline_frames": len(timeline),
        }
        
        return unique_classes, timeline, pipeline_stats
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the current detector."""
        info = self.manager.get_detector_info()
        info.update({
            "tracking_enabled": self.enable_tracking,
            "adaptive_sampling_enabled": self.enable_adaptive_sampling,
            "temporal_aggregation_enabled": self.enable_temporal_aggregation,
            "confidence_threshold": self.confidence,
            "effective_fps": self.sample_fps
        })
        return info


def detect_objects(
    video_path: str | Path,
    detector_type: str = "auto",
    confidence: float = 0.15,
    sample_fps: Optional[float] = None,
    prompt: Optional[str] = None,
    enable_tracking: bool = True,
    enable_adaptive_sampling: bool = True,
    enable_temporal_aggregation: bool = True
) -> Dict:
    """
    Simple function to detect objects in a video.
    
    Args:
        video_path: Path to video file
        detector_type: Detector type - 'auto', 'yolo-world', 'yolo', 'legacy'
        confidence: Detection confidence threshold
        sample_fps: Base frames per second to sample
        prompt: Text prompt for detection (e.g., "person, car, dog")
        enable_tracking: Enable object tracking across frames
        enable_adaptive_sampling: Enable motion-based adaptive sampling
        enable_temporal_aggregation: Enable voting across frames
        
    Returns:
        Dictionary with detection results
    """
    detector = VideoDetector(
        detector_type=detector_type,
        confidence=confidence,
        sample_fps=sample_fps,
        enable_tracking=enable_tracking,
        enable_adaptive_sampling=enable_adaptive_sampling,
        enable_temporal_aggregation=enable_temporal_aggregation
    )
    
    unique_classes, timeline, pipeline_stats = detector.detect_video(video_path, prompt=prompt)
    
    # Apply user corrections if any exist
    video_id = Path(video_path).stem
    try:
        from src.correction_applier import apply_corrections_to_timeline
        timeline = apply_corrections_to_timeline(timeline, video_id=video_id)
        # Rebuild unique classes from corrected timeline
        corrected_classes = set()
        for entry in timeline:
            for obj in entry.get("objects", []):
                corrected_classes.add(obj.get("class", ""))
        unique_classes = sorted(list(corrected_classes))
    except Exception as e:
        logger.warning(f"Correction application failed: {e}")
    
    return {
        "video": str(video_path),
        "detector": detector.get_detector_info(),
        "pipeline_stats": pipeline_stats,
        "classes": unique_classes,
        "total_classes": len(unique_classes),
        "timeline": timeline,
        "total_detections": len(timeline),
        "prompt": prompt
    }