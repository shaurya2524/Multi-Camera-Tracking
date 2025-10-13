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
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field


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


class AMCTracker:
    """
    Minimal AMC-based motion predictor that mimics the Kalman API.
    predict() -> returns np.array shape (4,1): [x, y, vx, vy]^T
    correct(measurement) -> accepts measurement (2x1) and does light bookkeeping.
    """

    def __init__(self, track_ref, max_absorbing_history: int = 5):
        self.track_ref = track_ref  # reference to GlobalTrack
        self.max_absorbing_history = max_absorbing_history
        # internal last prediction (for stability)
        self._last_pred = None

    def _build_candidates(self, last_pos, speed_est):
        """
        Build a small candidate grid around last_pos.
        Use step proportional to speed_est (but at least a minimum).
        Return list of (x,y) candidate coordinates.
        """
        min_step = 5.0
        step = max(min_step, speed_est * 1.5)
        offsets = [-step, 0.0, step]
        candidates = []
        for dx in offsets:
            for dy in offsets:
                candidates.append((last_pos[0] + dx, last_pos[1] + dy))
        # ensure unique
        uniq = []
        seen = set()
        for c in candidates:
            key = (round(c[0], 3), round(c[1], 3))
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        return uniq

    def predict(self):
        """
        Predict next [x,y,vx,vy] using a tiny AMC over candidate points.
        Returns shape (4,1) numpy array to match Kalman usage.
        """
        positions = self.track_ref.position_history
        if not positions:
            # no history -> return zeros
            arr = np.zeros((4, 1), dtype=np.float32)
            return arr

        last = positions[-1]
        # estimate speed from last two positions if available
        if len(positions) >= 2:
            prev = positions[-2]
            vx_est = last[0] - prev[0]
            vy_est = last[1] - prev[1]
            speed_est = np.hypot(vx_est, vy_est)
        else:
            vx_est, vy_est = 0.0, 0.0
            speed_est = 0.0

        # Build candidate transient nodes (small grid around last)
        candidates = self._build_candidates(last, speed_est)
        m = len(candidates)

        # Absorbing nodes: past few positions (considered 'background' targets to be avoided)
        absorbing_nodes = list(reversed(positions[:-0]))  # all historic positions
        # limit absorbing length
        absorbing_nodes = absorbing_nodes[-self.max_absorbing_history:]
        a = len(absorbing_nodes)
        if a == 0:
            # if no absorbing nodes (rare), fallback to returning last+velocity
            pred_x = last[0] + vx_est
            pred_y = last[1] + vy_est
            arr = np.array([[pred_x], [pred_y], [vx_est], [vy_est]], dtype=np.float32)
            self._last_pred = arr
            return arr

        # Compute transition weights (transient->transient and transient->absorbing)
        # Use Gaussian kernel on Euclidean distances
        sigma_t = max(1.0, speed_est + 1.0)
        sigma_a = max(1.0, speed_est + 1.0)

        # Q: m x m, R: m x a
        Q = np.zeros((m, m), dtype=np.float64)
        R = np.zeros((m, a), dtype=np.float64)
        for i in range(m):
            ci = np.array(candidates[i])
            # transient-transient
            for j in range(m):
                cj = np.array(candidates[j])
                dist = np.linalg.norm(ci - cj)
                Q[i, j] = np.exp(- (dist ** 2) / (2 * sigma_t ** 2))
            # transient-absorbing
            for k in range(a):
                ak = np.array(absorbing_nodes[k])
                dist = np.linalg.norm(ci - ak)
                R[i, k] = np.exp(- (dist ** 2) / (2 * sigma_a ** 2))

            # normalize row to sum to 1 (so it becomes a transition probability)
            row_sum = Q[i].sum() + R[i].sum()
            if row_sum > 0:
                Q[i] /= row_sum
                R[i] /= row_sum
            else:
                # rare: equally distribute
                Q[i] = np.ones(m, dtype=np.float64) / m

        # Q is now m x m transition among transient states
        # Fundamental matrix F = (I - Q)^-1
        I = np.eye(m, dtype=np.float64)
        try:
            F = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            # fallback to pseudoinverse for stability
            F = np.linalg.pinv(I - Q)

        # Standard absorption time (expected number of steps before absorption)
        # t = F * 1 (vector of ones), but we want modified absorption time:
        # compute expected number of visits to transient states before absorption:
        t = F.sum(axis=1)  # shape (m,)

        # Additionally compute expected visits to each absorbing node:
        # B = F * R  -> B[i,k] = expected visits from transient i to absorbing k
        B = F @ R  # (m x a)
        # Choose candidate with maximal combined score: e.g., t (staying in transient) minus closeness to absorbing nodes
        # We prefer candidates that visit foreground-like (i.e., avoid absorbing closish nodes)
        # A simple score: score = t - alpha * sum(B[:,k] * w_k) where w_k gives weight to absorbing nodes (use distance)
        # But to keep it simple and robust we'll choose candidate with maximum t (longer time before absorption => more foreground-like)
        best_idx = int(np.argmax(t))
        pred_x, pred_y = candidates[best_idx]

        # velocity estimate
        vx = pred_x - last[0]
        vy = pred_y - last[1]

        arr = np.array([[pred_x], [pred_y], [vx], [vy]], dtype=np.float32)
        self._last_pred = arr
        return arr

    def correct(self, measurement):
        """
        Called when a measurement is available. We don't have an internal filter to correct,
        because GlobalTrack.position_history is the canonical store. Keep as a no-op for now.
        """
        # No internal state which needs correction because we rely on position_history.
        return


@dataclass
class GlobalTrack:
    """Represents a global track across multiple cameras"""
    global_id: int
    camera_detections: Dict[int, "Detection"]   # camera_id -> Detection
    last_seen: float
    creation_time: float
    confidence_history: List[float]
    position_history: List[Tuple[float, float]]
    active_cameras: Set[int]

    # === NEW: Kalman filter field (ignored by dataclass repr/eq) ===
    kalman: AMCTracker = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Initialize AMC-based tracker for motion prediction"""
        # Replace Kalman filter with AMCTracker wrapper that implements predict() and correct()
        # so other parts of your system using kalman.predict()/correct() continue to work.
        self.kalman = AMCTracker(self)


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
            # Skip if track is from the same camera
            if detection.camera_id in track.active_cameras:
                continue

            if track.position_history:
                track_center = track.position_history[-1]

                # Use predicted position if available
                if hasattr(track, "predicted_position") and track.predicted_position is not None:
                    track_center = track.predicted_position

                # --- Adaptive distance threshold based on velocity ---
                # Get predicted velocity from AMC 'kalman' (predict returns [x,y,vx,vy])
                vx, vy = 0.0, 0.0
                if hasattr(track, "kalman") and track.kalman is not None:
                    prediction = track.kalman.predict()
                    vx, vy = float(prediction[2, 0]), float(prediction[3, 0])

                adaptive_threshold = self.distance_threshold + np.hypot(vx, vy) * 1.5

                distance = self.calculate_distance(ref_center, track_center)

                if distance <= adaptive_threshold:
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
        track.camera_detections[detection.camera_id] = detection
        track.active_cameras.add(detection.camera_id)
        track.last_seen = detection.timestamp

        # Update history
        track.confidence_history.append(detection.confidence)
        ref_center = self.transform_point_to_reference(detection.center, detection.camera_id)
        track.position_history.append(ref_center)

        # Limit history
        max_history = 50
        if len(track.confidence_history) > max_history:
            track.confidence_history = track.confidence_history[-max_history:]
            track.position_history = track.position_history[-max_history:]

        # === NEW: AMC-based 'correct' (kept for API compatibility) ===
        cx, cy = ref_center
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        track.kalman.correct(measurement)

    
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
        for global_id, track in self.global_tracks.items():
            frames_missing = self.frame_count - track.last_seen
            if frames_missing > 0 and frames_missing <= self.max_missing_frames:
                # Predict next position using AMCTracker
                prediction = track.kalman.predict()
                pred_x, pred_y = float(prediction[0]), float(prediction[1])
                # Store predicted position for matching
                track.predicted_position = (pred_x, pred_y)
            else:
                track.predicted_position = None


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
    def match_with_predictions(self, bbox, threshold: float = 50):
        """Try to match a new detection with predicted track positions"""
        x, y, w, h = bbox
        cx, cy = x + w//2, y + h//2

        for global_id, track in self.global_tracks.items():
            prediction = track.kalman.predict()
            pred_x, pred_y = int(prediction[0]), int(prediction[1])
            dist = np.hypot(cx - pred_x, cy - pred_y)
            if dist < threshold:
                return global_id
        return None

    def assign_existing_id(self, global_id: int, detection: "Detection", camera_id: int):
        """Force-assign detection to an existing track"""
        track = self.global_tracks[global_id]
        track.camera_detections[camera_id] = detection
        track.active_cameras.add(camera_id)
        track.last_seen = detection.timestamp

        # Correct AMC-based tracker (API compatible)
        cx, cy = self.transform_point_to_reference(detection.center, camera_id)
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        track.kalman.correct(measurement)



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
