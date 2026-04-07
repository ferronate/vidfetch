"""
Object tracking module for video detection.
Implements SORT (Simple Online Realtime Tracking) algorithm.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging

from src.utils import calculate_iou

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked object."""
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    age: int = 0
    hits: int = 0
    missed: int = 0
    
    def update(self, bbox: List[float], confidence: float):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.missed = 0
        self.age += 1
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.missed += 1
        self.age += 1
    
    def is_active(self, max_missed: int = 5) -> bool:
        """Check if track is still active."""
        return self.missed <= max_missed


class SORTTracker:
    """
    Simple Online Realtime Tracking (SORT) algorithm.
    Tracks objects across frames using IoU-based association.
    """
    
    def __init__(
        self,
        max_missed: int = 5,
        iou_threshold: float = 0.25,
        min_hits: int = 1
    ):
        """
        Initialize SORT tracker.
        
        Args:
            max_missed: Maximum frames a track can be missed before deletion
            iou_threshold: IoU threshold for matching detections to tracks
            min_hits: Minimum hits before track is confirmed
        """
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        return calculate_iou(bbox1, bbox2)
    
    def _match_detections_to_tracks(
        self,
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU.
        
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                iou_matrix[i, j] = self._calculate_iou(
                    det["bbox"],
                    self.tracks[track_id].bbox
                )
        
        # Use Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        
        # Convert to cost matrix (1 - IoU)
        cost_matrix = 1 - iou_matrix
        
        # Find optimal assignment
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        matched_pairs = []
        matched_det_indices = set()
        matched_track_indices = set()
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            iou = iou_matrix[det_idx, track_idx]
            if iou >= self.iou_threshold:
                matched_pairs.append((det_idx, track_ids[track_idx]))
                matched_det_indices.add(det_idx)
                matched_track_indices.add(track_idx)
        
        # Get unmatched detections and tracks
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in matched_track_indices]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections with 'bbox', 'class', 'confidence'
            
        Returns:
            List of tracked detections with 'track_id', 'bbox', 'class', 'confidence'
        """
        # Match detections to tracks
        matched_pairs, unmatched_dets, unmatched_tracks = \
            self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        for det_idx, track_id in matched_pairs:
            det = detections[det_idx]
            self.tracks[track_id].update(det["bbox"], det["confidence"])
        
        # Mark unmatched tracks as missed
        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=det["bbox"],
                class_name=det["class"],
                confidence=det["confidence"]
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        # Remove inactive tracks
        inactive_tracks = [
            track_id for track_id, track in self.tracks.items()
            if not track.is_active(self.max_missed)
        ]
        for track_id in inactive_tracks:
            del self.tracks[track_id]
        
        # Return confirmed tracks
        tracked_detections = []
        for track_id, track in self.tracks.items():
            if track.hits >= self.min_hits:
                tracked_detections.append({
                    "track_id": track_id,
                    "bbox": track.bbox,
                    "class": track.class_name,
                    "confidence": track.confidence,
                    "age": track.age,
                    "hits": track.hits
                })
        
        return tracked_detections
    
    def reset(self):
        """Reset all tracks."""
        self.tracks = {}
        self.next_id = 0
        logger.info("Tracker reset")


class TemporalAggregator:
    """
    Aggregates detections across frames using voting.
    Reduces flickering and improves consistency.
    """
    
    def __init__(self, window_size: int = 5, vote_threshold: float = 0.20):
        """
        Initialize temporal aggregator.
        
        Args:
            window_size: Number of frames to consider for voting
            vote_threshold: Ratio of frames object must appear in (0.0-1.0)
        """
        self.window_size = window_size
        self.vote_threshold = vote_threshold
        self.frame_history: List[Dict[int, Dict]] = []
    
    def add_frame(self, tracked_detections: List[Dict], timestamp: float):
        """
        Add a frame's detections to history.
        
        Args:
            tracked_detections: List of tracked detections
            timestamp: Frame timestamp
        """
        # Create dict keyed by track_id
        frame_dict = {}
        for det in tracked_detections:
            frame_dict[det["track_id"]] = det
        
        self.frame_history.append({
            "timestamp": timestamp,
            "detections": frame_dict
        })
        
        # Keep only recent history
        if len(self.frame_history) > self.window_size:
            self.frame_history.pop(0)
    
    def get_aggregated_detections(self) -> List[Dict]:
        """
        Get aggregated detections based on voting across frames.
        
        Returns:
            List of aggregated detections
        """
        if not self.frame_history:
            return []
        
        # Count appearances of each track_id
        track_counts: Dict[int, int] = {}
        track_detections: Dict[int, List[Dict]] = {}
        
        for frame in self.frame_history:
            for track_id, det in frame["detections"].items():
                track_counts[track_id] = track_counts.get(track_id, 0) + 1
                if track_id not in track_detections:
                    track_detections[track_id] = []
                track_detections[track_id].append(det)
        
        # Filter by vote threshold
        aggregated = []
        for track_id, count in track_counts.items():
            vote_ratio = count / len(self.frame_history)
            
            if vote_ratio >= self.vote_threshold:
                # Get most recent detection for this track
                recent_det = track_detections[track_id][-1]
                
                # Boost confidence based on consistency
                boosted_confidence = min(1.0, recent_det["confidence"] * (1 + vote_ratio))
                
                aggregated.append({
                    "track_id": track_id,
                    "class": recent_det["class"],
                    "confidence": boosted_confidence,
                    "bbox": recent_det["bbox"],
                    "vote_ratio": vote_ratio,
                    "appearances": count
                })
        
        return aggregated
    
    def reset(self):
        """Reset temporal aggregator."""
        self.frame_history = []