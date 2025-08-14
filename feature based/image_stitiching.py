import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch.nn.functional import cosine_similarity
from ultralytics import YOLO
from collections import defaultdict, deque
import time
from scipy.optimize import linear_sum_assignment

class PersonTracker:
    def __init__(self, similarity_threshold=0.7, max_age=30, feature_history_size=5):
        # Load models
        self.yolo_model = YOLO('yolov8n.pt')
        self.yolo_model.fuse()
        
        # Load ResNet50 for feature extraction
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = torch.nn.Identity()
        self.feature_extractor.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Tracking parameters
        self.similarity_threshold = similarity_threshold
        self.max_age = max_age
        self.feature_history_size = feature_history_size
        
        # Global tracking state
        self.global_id_counter = 0
        self.tracks = {}  # track_id -> Track object
        self.frame_count = 0
        
    def extract_feature(self, img_crop):
        """Extract feature vector from person crop"""
        try:
            if img_crop.size == 0 or img_crop.shape[0] < 10 or img_crop.shape[1] < 10:
                return None
            
            # Convert BGR to RGB
            img_crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img_crop_rgb).unsqueeze(0)
            
            with torch.no_grad():
                feature = self.feature_extractor(tensor).squeeze(0)
                # L2 normalize the feature
                feature = feature / torch.norm(feature)
            return feature
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        
        xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
        xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def compute_similarity_matrix(self, detections_left, detections_right):
        """Compute similarity matrix between left and right camera detections"""
        if not detections_left or not detections_right:
            return np.array([])
        
        sim_matrix = np.zeros((len(detections_left), len(detections_right)))
        
        for i, det_l in enumerate(detections_left):
            for j, det_r in enumerate(detections_right):
                # Feature similarity
                feat_sim = cosine_similarity(
                    det_l['feature'].unsqueeze(0), 
                    det_r['feature'].unsqueeze(0)
                ).item()
                
                # Height similarity (person should have similar height in both cameras)
                h_l = det_l['box'][3] - det_l['box'][1]
                h_r = det_r['box'][3] - det_r['box'][1]
                height_sim = min(h_l, h_r) / max(h_l, h_r) if max(h_l, h_r) > 0 else 0
                
                # Combined similarity
                sim_matrix[i, j] = feat_sim * 0.8 + height_sim * 0.2
        
        return sim_matrix
    
    def match_detections(self, detections_left, detections_right):
        """Match detections between cameras using Hungarian algorithm"""
        if not detections_left or not detections_right:
            return []
        
        sim_matrix = self.compute_similarity_matrix(detections_left, detections_right)
        
        # Convert similarity to cost (Hungarian algorithm minimizes)
        cost_matrix = 1 - sim_matrix
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        for i, j in zip(row_indices, col_indices):
            if sim_matrix[i, j] > self.similarity_threshold:
                matches.append((i, j, sim_matrix[i, j]))
        
        return matches
    
    def update_tracks(self, matched_pairs, detections_left, detections_right):
        """Update existing tracks and create new ones"""
        # Mark all tracks as not updated
        for track in self.tracks.values():
            track['updated'] = False
        
        # Update matched tracks
        for left_idx, right_idx, similarity in matched_pairs:
            det_l = detections_left[left_idx]
            det_r = detections_right[right_idx]
            
            # Try to find existing track
            matched_track_id = None
            best_sim = 0
            
            for track_id, track in self.tracks.items():
                if track['age'] > self.max_age:
                    continue
                
                # Compare with track's feature history
                track_sim = 0
                for hist_feat in track['feature_history']:
                    sim_l = cosine_similarity(det_l['feature'].unsqueeze(0), hist_feat.unsqueeze(0)).item()
                    sim_r = cosine_similarity(det_r['feature'].unsqueeze(0), hist_feat.unsqueeze(0)).item()
                    track_sim = max(track_sim, (sim_l + sim_r) / 2)
                
                if track_sim > best_sim and track_sim > self.similarity_threshold:
                    best_sim = track_sim
                    matched_track_id = track_id
            
            if matched_track_id is not None:
                # Update existing track
                track = self.tracks[matched_track_id]
                track['left_box'] = det_l['box']
                track['right_box'] = det_r['box']
                track['feature_history'].append((det_l['feature'] + det_r['feature']) / 2)
                if len(track['feature_history']) > self.feature_history_size:
                    track['feature_history'].popleft()
                track['age'] = 0
                track['updated'] = True
            else:
                # Create new track
                self.tracks[self.global_id_counter] = {
                    'id': self.global_id_counter,
                    'left_box': det_l['box'],
                    'right_box': det_r['box'],
                    'feature_history': deque([(det_l['feature'] + det_r['feature']) / 2], 
                                           maxlen=self.feature_history_size),
                    'age': 0,
                    'updated': True
                }
                self.global_id_counter += 1
        
        # Age unmatched tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if not track['updated']:
                track['age'] += 1
                if track['age'] > self.max_age:
                    tracks_to_remove.append(track_id)
        
        # Remove old tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def detect_persons(self, frame, x_offset=0):
        """Detect persons in frame and extract features"""
        results = self.yolo_model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            if int(box.cls[0].item()) != 0:  # Only person class
                continue
            
            conf = box.conf[0].item()
            if conf < 0.5:  # Confidence threshold
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Adjust coordinates if needed
            if x_offset > 0:
                x1 = max(0, x1 - x_offset)
                x2 = max(0, x2 - x_offset)
            
            # Extract crop and feature
            crop = frame[y1:y2, x1:x2]
            feature = self.extract_feature(crop)
            
            if feature is not None:
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'feature': feature,
                    'confidence': conf
                })
        
        return detections
    
    def draw_tracks(self, frame_left, frame_right):
        """Draw tracking results on frames"""
        display_left = frame_left.copy()
        display_right = frame_right.copy()
        
        for track_id, track in self.tracks.items():
            if track['age'] > 5:  # Don't draw very old tracks
                continue
            
            # Draw on left frame
            x1, y1, x2, y2 = track['left_box']
            color = (0, 255, 0) if track['age'] == 0 else (0, 255, 255)
            cv2.rectangle(display_left, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_left, f"ID: {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw on right frame
            x1, y1, x2, y2 = track['right_box']
            cv2.rectangle(display_right, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_right, f"ID: {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return display_left, display_right
    
    def process_frame(self, frame_left, frame_right):
        """Process a pair of frames"""
        self.frame_count += 1
        
        # Detect persons in both frames
        detections_left = self.detect_persons(frame_left)
        detections_right = self.detect_persons(frame_right)
        
        # Match detections between cameras
        matches = self.match_detections(detections_left, detections_right)
        
        # Update tracks
        self.update_tracks(matches, detections_left, detections_right)
        
        # Draw results
        display_left, display_right = self.draw_tracks(frame_left, frame_right)
        
        return display_left, display_right

def main():
    # Initialize tracker
    tracker = PersonTracker(similarity_threshold=0.7, max_age=30)
    
    # Video captures
    cap1 = cv2.VideoCapture('D:/ANITUM/cam1.mp4')
    cap2 = cv2.VideoCapture('D:/ANITUM/cam5.mp4')
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open video files")
        return
    
    # FPS calculation
    fps_counter = 0
    start_time = time.time()
    
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Resize frames
        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))
        
        # Process frames
        display_left, display_right = tracker.process_frame(frame1, frame2)
        
        # Calculate and display FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            elapsed = time.time() - start_time
            fps = fps_counter / elapsed
            print(f"FPS: {fps:.2f}, Active tracks: {len(tracker.tracks)}")
        
        # Display results
        combined_display = np.hstack((display_left, display_right))
        cv2.putText(combined_display, f"Active Tracks: {len(tracker.tracks)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Enhanced Multi-Camera Tracking", combined_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()