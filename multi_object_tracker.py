import cv2
import numpy as np
from collections import defaultdict
import json
import math

class MultiObjectTracker:
    def __init__(self, homography_path='homography_matrix.npy'):
        self.homography = np.load(homography_path)
        self.trackers = {}  # Store trackers for each object
        self.next_id = 0
        self.max_disappeared = 10  # Max frames before removing tracker
        self.max_distance = 100  # Max distance for matching objects
        
        # Object detection (using background subtraction)
        self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
    def detect_objects(self, frame, bg_subtractor, min_area=500):
        """Detect moving objects using background subtraction"""
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                objects.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': area
                })
        
        return objects, fg_mask
    
    def transform_point(self, point, camera_to_union=True):
        """Transform point between camera coordinate systems"""
        pt = np.array([[point[0], point[1]]], dtype=np.float32)
        
        if camera_to_union:
            # Transform from camera1 to union coordinate system
            transformed = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), self.homography)
        else:
            # Transform from union to camera1 coordinate system
            inv_homography = np.linalg.inv(self.homography)
            transformed = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), inv_homography)
        
        return tuple(transformed.reshape(-1, 2)[0].astype(int))
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update_trackers(self, objects_cam1, objects_cam2):
        """Update trackers with new detections from both cameras"""
        # Transform camera1 objects to union coordinate system
        union_objects = []
        
        # Add camera1 objects (transformed to union space)
        for obj in objects_cam1:
            transformed_center = self.transform_point(obj['center'])
            union_objects.append({
                'center': transformed_center,
                'bbox': obj['bbox'],
                'area': obj['area'],
                'camera': 1,
                'original_center': obj['center']
            })
        
        # Add camera2 objects (already in union space - camera2 is reference)
        for obj in objects_cam2:
            union_objects.append({
                'center': obj['center'],
                'bbox': obj['bbox'],
                'area': obj['area'],
                'camera': 2,
                'original_center': obj['center']
            })
        
        # Match objects with existing trackers
        matched_ids = set()
        for obj in union_objects:
            best_match_id = None
            best_distance = float('inf')
            
            for tracker_id, tracker in self.trackers.items():
                if tracker_id in matched_ids:
                    continue
                    
                distance = self.calculate_distance(obj['center'], tracker['last_position'])
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match_id = tracker_id
            
            if best_match_id is not None:
                # Update existing tracker
                self.trackers[best_match_id]['last_position'] = obj['center']
                self.trackers[best_match_id]['disappeared'] = 0
                self.trackers[best_match_id]['camera'] = obj['camera']
                self.trackers[best_match_id]['bbox'] = obj['bbox']
                self.trackers[best_match_id]['original_center'] = obj['original_center']
                matched_ids.add(best_match_id)
            else:
                # Create new tracker
                self.trackers[self.next_id] = {
                    'last_position': obj['center'],
                    'disappeared': 0,
                    'camera': obj['camera'],
                    'bbox': obj['bbox'],
                    'original_center': obj['original_center']
                }
                self.next_id += 1
        
        # Update disappeared count for unmatched trackers
        to_remove = []
        for tracker_id, tracker in self.trackers.items():
            if tracker_id not in matched_ids:
                tracker['disappeared'] += 1
                if tracker['disappeared'] > self.max_disappeared:
                    to_remove.append(tracker_id)
        
        # Remove disappeared trackers
        for tracker_id in to_remove:
            del self.trackers[tracker_id]
    
    def draw_tracks(self, frame, camera_id, union_frame=None):
        """Draw tracking information on frame"""
        for tracker_id, tracker in self.trackers.items():
            if tracker['camera'] == camera_id:
                # Draw on original camera frame
                center = tracker['original_center']
                bbox = tracker['bbox']
                
                # Draw bounding box
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Draw ID
                cv2.putText(frame, f'ID: {tracker_id}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw on union frame if provided
        if union_frame is not None:
            for tracker_id, tracker in self.trackers.items():
                center = tracker['last_position']
                
                # Draw center point
                cv2.circle(union_frame, center, 8, (0, 0, 255), -1)
                
                # Draw ID
                cv2.putText(union_frame, f'ID: {tracker_id}', 
                           (center[0] + 10, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw camera info
                cam_color = (255, 0, 0) if tracker['camera'] == 1 else (0, 255, 255)
                cv2.putText(union_frame, f'Cam{tracker["camera"]}', 
                           (center[0] + 10, center[1] + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, cam_color, 1)
    
    def get_tracking_stats(self):
        """Get current tracking statistics"""
        active_trackers = len(self.trackers)
        cam1_objects = sum(1 for t in self.trackers.values() if t['camera'] == 1)
        cam2_objects = sum(1 for t in self.trackers.values() if t['camera'] == 2)
        
        return {
            'total_objects': active_trackers,
            'cam1_objects': cam1_objects,
            'cam2_objects': cam2_objects
        }

# Example usage
if __name__ == "__main__":
    tracker = MultiObjectTracker()
    
    cap1 = cv2.VideoCapture('video1.mp4')
    cap2 = cv2.VideoCapture('video2.mp4')
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Detect objects in both cameras
        objects1, mask1 = tracker.detect_objects(frame1, tracker.bg_subtractor1)
        objects2, mask2 = tracker.detect_objects(frame2, tracker.bg_subtractor2)
        
        # Update trackers
        tracker.update_trackers(objects1, objects2)
        
        # Draw tracking results
        tracker.draw_tracks(frame1, 1)
        tracker.draw_tracks(frame2, 2)
        
        # Create union frame for visualization
        h2, w2 = frame2.shape[:2]
        union_w, union_h = w2 + 200, h2 + 200
        union_frame = np.zeros((union_h, union_w, 3), dtype=np.uint8)
        
        # Place frame2 (reference)
        union_frame[:h2, :w2] = frame2
        
        # Warp and blend frame1
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
        cv2.imshow('Camera 1 - Tracking', frame1)
        cv2.imshow('Camera 2 - Tracking', frame2)
        cv2.imshow('Union - Multi-Camera Tracking', union_frame)
        
        # Print stats
        stats = tracker.get_tracking_stats()
        print(f"\rTracking: {stats['total_objects']} objects | "
              f"Cam1: {stats['cam1_objects']} | Cam2: {stats['cam2_objects']}", end='')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()