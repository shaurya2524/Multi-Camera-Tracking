import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import time

class KalmanFilter:
    """Simple Kalman filter for tracking object motion"""
    def __init__(self, bbox):
        """Initialize Kalman filter with bounding box [x1, y1, x2, y2]"""
        self.dt = 1.0
        
        # State vector: [x, y, vx, vy, w, h, vw, vh]
        # x, y: center coordinates
        # vx, vy: velocity
        # w, h: width, height
        # vw, vh: size change rate
        
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        self.x = np.array([x, y, 0, 0, w, h, 0, 0], dtype=np.float32)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0, 0, 0, 0, 0],
            [0, 1, 0, self.dt, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Covariance matrices
        self.P = np.eye(8, dtype=np.float32) * 1000  # Initial uncertainty
        self.Q = np.eye(8, dtype=np.float32) * 0.1   # Process noise
        self.R = np.eye(4, dtype=np.float32) * 10    # Measurement noise
        
    def predict(self):
        """Predict next state"""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # Convert to bbox format
        x, y, w, h = self.x[0], self.x[1], self.x[4], self.x[5]
        return [x - w/2, y - h/2, x + w/2, y + h/2]
    
    def update(self, bbox):
        """Update with measurement"""
        # Convert bbox to measurement
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([x, y, w, h], dtype=np.float32)
        
        # Kalman update
        y_residual = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y_residual)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

class Track:
    """Individual track for a person"""
    def __init__(self, track_id: int, detection: Dict, features: np.ndarray):
        self.track_id = track_id
        self.kalman = KalmanFilter(detection['bbox'])
        self.features = [features]
        self.detections = [detection]
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.occlusion_count = 0
        self.is_confirmed = False
        self.state = 'tentative'  # tentative, confirmed, deleted
        
    def predict(self):
        """Predict next position"""
        if self.time_since_update > 0:
            self.occlusion_count += 1
        return self.kalman.predict()
    
    def update(self, detection: Dict, features: np.ndarray):
        """Update track with new detection"""
        self.kalman.update(detection['bbox'])
        self.features.append(features)
        self.detections.append(detection)
        
        # Keep only recent features (for efficiency)
        if len(self.features) > 100:
            self.features = self.features[-100:]
            self.detections = self.detections[-100:]
        
        self.hits += 1
        self.time_since_update = 0
        self.occlusion_count = 0
        
        # Confirm track if it has enough hits
        if self.hits >= 3 and self.state == 'tentative':
            self.state = 'confirmed'
            self.is_confirmed = True
    
    def mark_missed(self):
        """Mark track as missed"""
        self.time_since_update += 1
        self.age += 1
        
        # Delete track if missed for too long
        if self.time_since_update > 30:
            self.state = 'deleted'
    
    def get_current_bbox(self):
        """Get current bounding box"""
        if self.detections:
            return self.detections[-1]['bbox']
        return self.predict()
    
    def get_average_feature(self):
        """Get average feature vector"""
        if self.features:
            return np.mean(self.features[-10:], axis=0)  # Use last 10 features
        return None

class HybridTracker:
    """Hybrid tracker combining motion prediction and appearance matching"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0
        
    def compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / (area1 + area2 - intersection)
    
    def compute_distance(self, bbox1, bbox2):
        """Compute euclidean distance between bbox centers"""
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def compute_appearance_similarity(self, features1, features2):
        """Compute cosine similarity between feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        
        # Normalize features
        features1 = features1 / (np.linalg.norm(features1) + 1e-8)
        features2 = features2 / (np.linalg.norm(features2) + 1e-8)
        
        return np.dot(features1, features2)
    
    def associate_detections_to_tracks(self, detections, detection_features):
        """Associate detections to existing tracks"""
        if not self.tracks or not detections:
            return [], list(range(len(detections)))
        
        # Predict track positions
        predicted_bboxes = []
        for track in self.tracks:
            predicted_bboxes.append(track.predict())
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t, track in enumerate(self.tracks):
            for d, detection in enumerate(detections):
                # Motion similarity (IoU)
                iou = self.compute_iou(predicted_bboxes[t], detection['bbox'])
                
                # Distance similarity
                distance = self.compute_distance(predicted_bboxes[t], detection['bbox'])
                distance_sim = max(0, 1 - distance / self.max_distance)
                
                # Appearance similarity
                track_features = track.get_average_feature()
                if track_features is not None and detection_features[d] is not None:
                    appearance_sim = self.compute_appearance_similarity(
                        track_features, detection_features[d]
                    )
                else:
                    appearance_sim = 0.0
                
                # Combined cost (lower is better)
                motion_cost = 1 - (0.7 * iou + 0.3 * distance_sim)
                appearance_cost = 1 - appearance_sim
                
                # Hybrid cost with occlusion handling
                if track.occlusion_count > 5:
                    # Rely more on appearance when occluded
                    cost_matrix[t, d] = 0.3 * motion_cost + 0.7 * appearance_cost
                else:
                    # Rely more on motion when visible
                    cost_matrix[t, d] = 0.7 * motion_cost + 0.3 * appearance_cost
        
        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out assignments with high cost
        matches = []
        unmatched_detections = list(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 0.8:  # Threshold for valid match
                matches.append((row, col))
                unmatched_detections.remove(col)
        
        return matches, unmatched_detections
    
    def update(self, detections: List[Dict], detection_features: List[np.ndarray]):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Associate detections to tracks
        matches, unmatched_detections = self.associate_detections_to_tracks(
            detections, detection_features
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], detection_features[det_idx])
        
        # Mark unmatched tracks as missed
        matched_track_indices = [m[0] for m in matches]
        for i, track in enumerate(self.tracks):
            if i not in matched_track_indices:
                track.mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = Track(
                self.next_id, 
                detections[det_idx], 
                detection_features[det_idx]
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove deleted tracks
        self.tracks = [track for track in self.tracks if track.state != 'deleted']
        
        return self.get_tracklets()
    
    def get_tracklets(self):
        """Get current tracklets"""
        tracklets = []
        for track in self.tracks:
            if track.is_confirmed:
                tracklets.append({
                    'track_id': track.track_id,
                    'bbox': track.get_current_bbox(),
                    'features': track.get_average_feature(),
                    'age': track.age,
                    'hits': track.hits,
                    'time_since_update': track.time_since_update
                })
        return tracklets