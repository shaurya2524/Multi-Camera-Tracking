#!/usr/bin/env python3
"""
Global ID Manager for Multi-Camera Tracking

This module handles the creation and management of global IDs by merging
close detections from different cameras into single global identities.

Author: Multi-Camera Tracking System
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import cv2


@dataclass
class Detection:
    """Represents a detection from a camera"""
    camera_id: int
    local_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[float, float]      # x, y center point
    confidence: float
    class_id: int
    timestamp: float


@dataclass
class GlobalTrack:
    """Represents a global track across multiple cameras"""
    global_id: int
    camera_detections: Dict[int, Detection]  # camera_id -> Detection
    last_seen: float
    creation_time: float
    confidence_history: List[float]
    position_history: List[Tuple[float, float]]
    active_cameras: Set[int]


class GlobalIDManager:
    """
    Manages global IDs for multi-camera tracking system.
    
    This class handles:
    - Merging close detections from different cameras
    - Maintaining global track continuity
    - Managing track lifecycle (creation, update, deletion)
    """
    
    def __init__(self, 
                 distance_threshold: float = 50.0,
                 confidence_threshold: float = 0.5,
                 max_missing_frames: int = 30,
                 min_track_length: int = 5,
                 merge_iou_threshold: float = 0.3):
        """
        Initialize Global ID Manager
        
        Args:
            distance_threshold: Maximum distance for merging detections (pixels)
            confidence_threshold: Minimum confidence for valid detections
            max_missing_frames: Maximum frames a track can be missing before deletion
            min_track_length: Minimum track length before considering it stable
            merge_iou_threshold: IoU threshold for bbox overlap consideration
        """
        self.distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold
        self.max_missing_frames = max_missing_frames
        self.min_track_length = min_track_length
        self.merge_iou_threshold = merge_iou_threshold
        
        # Track management
        self.global_tracks: Dict[int, GlobalTrack] = {}
        self.next_global_id = 1
        self.frame_count = 0
        
        # Camera mapping and homography
        self.camera_homographies: Dict[int, np.ndarray] = {}
        self.reference_camera = 2  # Camera 2 as reference (from your existing pipeline)
        
    def set_homography(self, camera_id: int, homography_matrix: np.ndarray):
        """Set homography matrix for camera to reference camera transformation"""
        self.camera_homographies[camera_id] = homography_matrix
    
    def transform_point_to_reference(self, point: Tuple[float, float], camera_id: int) -> Tuple[float, float]:
        """Transform point from camera coordinate to reference coordinate system"""
        if camera_id == self.reference_camera or camera_id not in self.camera_homographies:
            return point
        
        # Transform point using homography
        point_homogeneous = np.array([[point[0], point[1], 1.0]], dtype=np.float32).T
        transformed = self.camera_homographies[camera_id] @ point_homogeneous
        
        # Convert back to Cartesian coordinates
        x = transformed[0, 0] / transformed[2, 0]
        y = transformed[1, 0] / transformed[2, 0]
        
        return (float(x), float(y))
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def create_detection(self, camera_id: int, local_id: int, bbox: Tuple[int, int, int, int], 
                        confidence: float, class_id: int = 0) -> Detection:
        """Create a Detection object from raw detection data"""
        x, y, w, h = bbox
        center = (x + w/2, y + h/2)
        
        return Detection(
            camera_id=camera_id,
            local_id=local_id,
            bbox=bbox,
            center=center,
            confidence=confidence,
            class_id=class_id,
            timestamp=self.frame_count
        )
    
    def find_matching_tracks(self, detection: Detection) -> List[Tuple[int, float]]:
        """Find existing global tracks that could match this detection"""
        matches = []
        
        # Transform detection center to reference coordinates
        ref_center = self.transform_point_to_reference(detection.center, detection.camera_id)
        
        for global_id, track in self.global_tracks.items():
            # Skip if track is from the same camera (shouldn't happen in normal flow)
            if detection.camera_id in track.active_cameras:
                continue
            
            # Get the most recent position from track history
            if track.position_history:
                track_center = track.position_history[-1]
                distance = self.calculate_distance(ref_center, track_center)
                
                if distance <= self.distance_threshold:
                    matches.append((global_id, distance))
        
        # Sort by distance (closest first)
        matches.sort(key=lambda x: x[1])
        return matches
    
    def create_new_global_track(self, detection: Detection) -> GlobalTrack:
        """Create a new global track from detection"""
        global_id = self.next_global_id
        self.next_global_id += 1
        
        ref_center = self.transform_point_to_reference(detection.center, detection.camera_id)
        
        track = GlobalTrack(
            global_id=global_id,
            camera_detections={detection.camera_id: detection},
            last_seen=detection.timestamp,
            creation_time=detection.timestamp,
            confidence_history=[detection.confidence],
            position_history=[ref_center],
            active_cameras={detection.camera_id}
        )
        
        return track
    
    def update_global_track(self, track: GlobalTrack, detection: Detection):
        """Update existing global track with new detection"""
        # Update detection for this camera
        track.camera_detections[detection.camera_id] = detection
        track.active_cameras.add(detection.camera_id)
        track.last_seen = detection.timestamp
        
        # Update history
        track.confidence_history.append(detection.confidence)
        ref_center = self.transform_point_to_reference(detection.center, detection.camera_id)
        track.position_history.append(ref_center)
        
        # Limit history length to prevent memory issues
        max_history = 50
        if len(track.confidence_history) > max_history:
            track.confidence_history = track.confidence_history[-max_history:]
            track.position_history = track.position_history[-max_history:]
    
    def process_detections(self, camera_detections: Dict[int, List[Tuple]]) -> Dict[int, GlobalTrack]:
        """
        Process detections from all cameras and return updated global tracks
        
        Args:
            camera_detections: Dict[camera_id, List[(local_id, bbox, confidence)]]
            
        Returns:
            Dict[global_id, GlobalTrack]: Updated global tracks
        """
        self.frame_count += 1
        
        # Convert raw detections to Detection objects
        all_detections = []
        for camera_id, detections in camera_detections.items():
            for local_id, bbox, confidence in detections:
                if confidence >= self.confidence_threshold:
                    detection = self.create_detection(camera_id, local_id, bbox, confidence)
                    all_detections.append(detection)
        
        # Reset active cameras for all tracks
        for track in self.global_tracks.values():
            track.active_cameras.clear()
        
        # Process each detection
        unmatched_detections = []
        
        for detection in all_detections:
            matches = self.find_matching_tracks(detection)
            
            if matches:
                # Match with closest existing track
                best_match_id, _ = matches[0]
                self.update_global_track(self.global_tracks[best_match_id], detection)
            else:
                unmatched_detections.append(detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = self.create_new_global_track(detection)
            self.global_tracks[new_track.global_id] = new_track
        
        # Clean up old tracks
        self._cleanup_old_tracks()
        
        return self.global_tracks.copy()
    
    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been seen for too long"""
        current_time = self.frame_count
        tracks_to_remove = []
        
        for global_id, track in self.global_tracks.items():
            frames_missing = current_time - track.last_seen
            
            if frames_missing > self.max_missing_frames:
                tracks_to_remove.append(global_id)
        
        for global_id in tracks_to_remove:
            del self.global_tracks[global_id]
    
    def get_global_id_for_local(self, camera_id: int, local_id: int) -> Optional[int]:
        """Get global ID for a local track ID from specific camera"""
        for global_id, track in self.global_tracks.items():
            if (camera_id in track.camera_detections and 
                track.camera_detections[camera_id].local_id == local_id):
                return global_id
        return None
    
    def get_active_global_tracks(self) -> Dict[int, GlobalTrack]:
        """Get all currently active global tracks"""
        active_tracks = {}
        current_time = self.frame_count
        
        for global_id, track in self.global_tracks.items():
            if current_time - track.last_seen <= 5:  # Recently seen
                active_tracks[global_id] = track
        
        return active_tracks
    
    def draw_global_tracks(self, frame: np.ndarray, camera_id: Optional[int] = None) -> np.ndarray:
        """
        Draw global tracks on frame
        
        Args:
            frame: Image frame to draw on
            camera_id: If specified, only draw tracks visible in this camera
            
        Returns:
            Frame with global tracks drawn
        """
        result_frame = frame.copy()
        active_tracks = self.get_active_global_tracks()
        
        for global_id, track in active_tracks.items():
            # If camera_id specified, only draw if track is in that camera
            if camera_id is not None and camera_id not in track.camera_detections:
                continue
            
            # Choose color based on global ID
            color = self._get_track_color(global_id)
            
            if camera_id is not None:
                # Draw for specific camera
                detection = track.camera_detections[camera_id]
                x, y, w, h = detection.bbox
                
                # Draw bounding box
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw global ID
                label = f"G{global_id}"
                cv2.putText(result_frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                # Draw for union view - use position history
                if track.position_history:
                    center = track.position_history[-1]
                    center_int = (int(center[0]), int(center[1]))
                    
                    # Draw track point
                    cv2.circle(result_frame, center_int, 8, color, -1)
                    cv2.circle(result_frame, center_int, 10, (255, 255, 255), 2)
                    
                    # Draw global ID
                    label = f"G{global_id}"
                    cv2.putText(result_frame, label, 
                               (center_int[0] - 20, center_int[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw track history
                    if len(track.position_history) > 1:
                        points = [(int(p[0]), int(p[1])) for p in track.position_history[-10:]]
                        for i in range(1, len(points)):
                            cv2.line(result_frame, points[i-1], points[i], color, 2)
        
        return result_frame
    
    def _get_track_color(self, global_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID"""
        # Simple hash-based color generation
        np.random.seed(global_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        active_tracks = self.get_active_global_tracks()
        
        stats = {
            'total_global_tracks': len(self.global_tracks),
            'active_global_tracks': len(active_tracks),
            'frame_count': self.frame_count,
            'next_global_id': self.next_global_id,
            'tracks_per_camera': {}
        }
        
        # Count tracks per camera
        for camera_id in [1, 2]:  # Assuming 2 cameras
            count = sum(1 for track in active_tracks.values() 
                       if camera_id in track.active_cameras)
            stats['tracks_per_camera'][f'camera_{camera_id}'] = count
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Create manager
    manager = GlobalIDManager(distance_threshold=30.0)
    
    # Set homography (example - you'll load from your existing system)
    # manager.set_homography(1, np.load('homography_matrix.npy'))
    
    # Example detection processing
    detections = {
        1: [(1, (100, 100, 50, 80), 0.9), (2, (200, 150, 45, 75), 0.8)],
        2: [(1, (110, 105, 48, 78), 0.85), (2, (300, 200, 50, 85), 0.9)]
    }
    
    global_tracks = manager.process_detections(detections)
    
    print("Global Tracks:")
    for global_id, track in global_tracks.items():
        print(f"Global ID {global_id}: Cameras {list(track.active_cameras)}")
    
    print("\nStatistics:", manager.get_statistics())