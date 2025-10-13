import cv2
import numpy as np
from collections import defaultdict
import json

class KalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        
        self.last_position = None
        self.disappeared = 0
        self.predictions = []
        
    def predict(self):
        """Predict next position"""
        prediction = self.kalman.predict()
        predicted_pos = (int(prediction[0]), int(prediction[1]))
        self.predictions.append(predicted_pos)
        if len(self.predictions) > 10:  # Keep last 10 predictions
            self.predictions.pop(0)
        return predicted_pos
    
    def update(self, measurement):
        """Update with new measurement"""
        self.kalman.correct(np.array([[measurement[0]], [measurement[1]]], dtype=np.float32))
        self.last_position = measurement
        self.disappeared = 0

class EnhancedMultiTracker:
    def __init__(self, homography_path='homography_matrix.npy'):
        self.homography = np.load(homography_path)
        self.trackers = {}  # Store KalmanTracker objects
        self.next_id = 0
        self.max_disappeared = 15
        self.max_distance = 80
        
        # Enhanced object detection
        self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, history=500, varThreshold=50)
        self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, history=500, varThreshold=50)
    
    def detect_objects(self, frame, bg_subtractor, min_area=800):
        """Enhanced object detection with filtering"""
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Enhanced morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Additional filtering
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (avoid very thin objects)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5.0:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    objects.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        return objects, fg_mask
    
    def transform_point(self, point):
        """Transform point using homography"""
        pt = np.array([[point[0], point[1]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), self.homography)
        return tuple(transformed.reshape(-1, 2)[0].astype(int))
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update_trackers(self, objects_cam1, objects_cam2):
        """Update trackers with Kalman filter predictions"""
        # Get predictions from existing trackers
        predictions = {}
        for tracker_id, tracker in self.trackers.items():
            predictions[tracker_id] = tracker.predict()
        
        # Prepare union objects
        union_objects = []
        
        # Transform camera1 objects
        for obj in objects_cam1:
            transformed_center = self.transform_point(obj['center'])
            union_objects.append({
                'center': transformed_center,
                'bbox': obj['bbox'],
                'area': obj['area'],
                'camera': 1,
                'original_center': obj['center']
            })
        
        # Add camera2 objects
        for obj in objects_cam2:
            union_objects.append({
                'center': obj['center'],
                'bbox': obj['bbox'],
                'area': obj['area'],
                'camera': 2,
                'original_center': obj['center']
            })
        
        # Match objects with trackers using both current position and prediction
        matched_ids = set()
        for obj in union_objects:
            best_match_id = None
            best_score = float('inf')
            
            for tracker_id, tracker in self.trackers.items():
                if tracker_id in matched_ids:
                    continue
                
                # Calculate distance to prediction
                pred_distance = self.calculate_distance(obj['center'], predictions[tracker_id])
                
                # Calculate distance to last known position
                if tracker.last_position:
                    pos_distance = self.calculate_distance(obj['center'], tracker.last_position)
                    # Combine both distances with weights
                    combined_score = 0.7 * pred_distance + 0.3 * pos_distance
                else:
                    combined_score = pred_distance
                
                if combined_score < self.max_distance and combined_score < best_score:
                    best_score = combined_score
                    best_match_id = tracker_id
            
            if best_match_id is not None:
                # Update existing tracker
                self.trackers[best_match_id].update(obj['center'])
                self.trackers[best_match_id].camera = obj['camera']
                self.trackers[best_match_id].bbox = obj['bbox']
                self.trackers[best_match_id].original_center = obj['original_center']
                matched_ids.add(best_match_id)
            else:
                # Create new tracker
                new_tracker = KalmanTracker()
                # Initialize Kalman filter
                new_tracker.kalman.statePre = np.array([obj['center'][0], obj['center'][1], 0, 0], dtype=np.float32)
                new_tracker.kalman.statePost = np.array([obj['center'][0], obj['center'][1], 0, 0], dtype=np.float32)
                new_tracker.update(obj['center'])
                new_tracker.camera = obj['camera']
                new_tracker.bbox = obj['bbox']
                new_tracker.original_center = obj['original_center']
                
                self.trackers[self.next_id] = new_tracker
                self.next_id += 1
        
        # Update disappeared count for unmatched trackers
        to_remove = []
        for tracker_id, tracker in self.trackers.items():
            if tracker_id not in matched_ids:
                tracker.disappeared += 1
                if tracker.disappeared > self.max_disappeared:
                    to_remove.append(tracker_id)
        
        # Remove disappeared trackers
        for tracker_id in to_remove:
            del self.trackers[tracker_id]
    
    def draw_tracks(self, frame, camera_id, union_frame=None):
        """Draw tracking with predictions and trails"""
        for tracker_id, tracker in self.trackers.items():
            if hasattr(tracker, 'camera') and tracker.camera == camera_id:
                # Draw on original camera frame
                center = tracker.original_center
                bbox = tracker.bbox
                
                # Draw bounding box
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Draw ID
                cv2.putText(frame, f'ID: {tracker_id}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw on union frame with predictions
        if union_frame is not None:
            for tracker_id, tracker in self.trackers.items():
                if tracker.last_position:
                    center = tracker.last_position
                    
                    # Draw prediction trail
                    for i, pred_pos in enumerate(tracker.predictions):
                        alpha = (i + 1) / len(tracker.predictions)
                        color = (int(255 * alpha), int(100 * alpha), 0)
                        cv2.circle(union_frame, pred_pos, 3, color, -1)
                    
                    # Draw current position
                    cv2.circle(union_frame, center, 8, (0, 0, 255), -1)
                    
                    # Draw ID
                    cv2.putText(union_frame, f'ID: {tracker_id}', 
                               (center[0] + 10, center[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw camera info
                    if hasattr(tracker, 'camera'):
                        cam_color = (255, 0, 0) if tracker.camera == 1 else (0, 255, 255)
                        cv2.putText(union_frame, f'Cam{tracker.camera}', 
                                   (center[0] + 10, center[1] + 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, cam_color, 1)
    
    def save_tracking_data(self, filename='tracking_data.json'):
        """Save tracking data to JSON file"""
        data = {}
        for tracker_id, tracker in self.trackers.items():
            if tracker.last_position:
                data[tracker_id] = {
                    'position': tracker.last_position,
                    'camera': getattr(tracker, 'camera', 0),
                    'predictions': tracker.predictions
                }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

# Example usage with enhanced features
if __name__ == "__main__":
    tracker = EnhancedMultiTracker()
    
    cap1 = cv2.VideoCapture('video1.mp4')
    cap2 = cv2.VideoCapture('video2.mp4')
    
    frame_count = 0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        frame_count += 1
        
        # Detect objects
        objects1, mask1 = tracker.detect_objects(frame1, tracker.bg_subtractor1)
        objects2, mask2 = tracker.detect_objects(frame2, tracker.bg_subtractor2)
        
        # Update trackers
        tracker.update_trackers(objects1, objects2)
        
        # Draw tracking results
        tracker.draw_tracks(frame1, 1)
        tracker.draw_tracks(frame2, 2)
        
        # Create union frame
        h2, w2 = frame2.shape[:2]
        union_w, union_h = w2 + 200, h2 + 200
        union_frame = np.zeros((union_h, union_w, 3), dtype=np.uint8)
        
        union_frame[:h2, :w2] = frame2
        warped1 = cv2.warpPerspective(frame1, tracker.homography, (union_w, union_h))
        mask1_warped = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY)
        mask1_warped = (mask1_warped > 0).astype(np.uint8) * 255
        
        for c in range(3):
            union_frame[:, :, c] = np.where(mask1_warped > 0, 
                                           (union_frame[:, :, c] * 0.5 + warped1[:, :, c] * 0.5).astype(np.uint8),
                                           union_frame[:, :, c])
        
        # Draw tracks on union frame
        tracker.draw_tracks(None, None, union_frame)
        
        # Display results
        cv2.imshow('Camera 1 - Enhanced Tracking', frame1)
        cv2.imshow('Camera 2 - Enhanced Tracking', frame2)
        cv2.imshow('Union - Kalman Tracking', union_frame)
        
        # Save tracking data periodically
        if frame_count % 30 == 0:
            tracker.save_tracking_data()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()