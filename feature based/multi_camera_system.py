import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import threading
import queue
import os
import csv

from detection import PersonDetector
from hybrid_tracker import HybridTracker
from feature_extractor import FeatureExtractorFactory
from cross_camera_association import CrossCameraAssociation

class MultiCameraTrackingSystem:
    """Complete multi-camera tracking system"""
    
    def __init__(self, camera_configs: List[Dict], feature_extractor_type='resnet50'):
        """
        Initialize multi-camera tracking system
        
        Args:
            camera_configs: List of camera configurations
            feature_extractor_type: Type of feature extractor to use
        """
        self.camera_configs = camera_configs
        self.num_cameras = len(camera_configs)
        
        # Initialize components
        self.detector = PersonDetector(model_size='n', confidence_threshold=0.5)
        self.feature_extractor = FeatureExtractorFactory.create_extractor(feature_extractor_type)
        self.cross_camera_association = CrossCameraAssociation()
        
        # Single-camera trackers for each camera
        self.single_trackers = {}
        for i, config in enumerate(camera_configs):
            self.single_trackers[i] = HybridTracker()
        
        # Video captures
        self.video_captures = {}
        self.frame_queues = {}
        self.result_queues = {}
        
        # Threading
        self.camera_threads = {}
        self.processing_threads = {}
        self.running = False
        
        # Results storage
        self.latest_results = {}
        self.global_tracks = {}
        
        # Performance metrics
        self.fps_counters = defaultdict(lambda: {'count': 0, 'start_time': time.time()})
        
    def initialize_cameras(self):
        """Initialize all cameras"""
        self.active_cameras = []
        for i, config in enumerate(self.camera_configs):
            # Initialize video capture
            source = config.get('source', 0)
            self.video_captures[i] = cv2.VideoCapture(source)
            if not self.video_captures[i].isOpened():
                print(f"[ERROR] Camera {i} failed to open source: {source}")
            else:
                print(f"[DEBUG] Camera {i} initialized: {config}")
                self.active_cameras.append(i)
            # Set camera properties if specified
            if 'width' in config:
                self.video_captures[i].set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
            if 'height' in config:
                self.video_captures[i].set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
            if 'fps' in config:
                self.video_captures[i].set(cv2.CAP_PROP_FPS, config['fps'])
            # Initialize queues
            self.frame_queues[i] = queue.Queue(maxsize=5)
            self.result_queues[i] = queue.Queue(maxsize=10)
        print(f"[INFO] Active cameras: {self.active_cameras}")
        if not self.active_cameras:
            print("[FATAL] No cameras could be initialized. Exiting.")
            exit(1)
    
    def start_tracking(self):
        """Start multi-camera tracking"""
        self.running = True
        print(f"[INFO] Starting tracking for cameras: {list(self.video_captures.keys())}")
        # Start camera capture threads
        for camera_id in self.active_cameras:
            print(f"[DEBUG] Starting camera capture thread for camera {camera_id}")
            self.camera_threads[camera_id] = threading.Thread(
                target=self._camera_capture_loop,
                args=(camera_id,)
            )
            self.camera_threads[camera_id].daemon = True
            self.camera_threads[camera_id].start()
        # Start processing threads
        for camera_id in self.active_cameras:
            print(f"[DEBUG] Starting processing thread for camera {camera_id}")
            self.processing_threads[camera_id] = threading.Thread(
                target=self._processing_loop,
                args=(camera_id,)
            )
            self.processing_threads[camera_id].daemon = True
            self.processing_threads[camera_id].start()
        # Start cross-camera association thread
        self.association_thread = threading.Thread(target=self._association_loop)
        self.association_thread.daemon = True
        self.association_thread.start()
        print("[INFO] Multi-camera tracking system started")
    
    def stop_tracking(self):
        """Stop multi-camera tracking"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.camera_threads.values():
            thread.join(timeout=1)
        
        for thread in self.processing_threads.values():
            thread.join(timeout=1)
        
        if hasattr(self, 'association_thread'):
            self.association_thread.join(timeout=1)
        
        # Release video captures
        for cap in self.video_captures.values():
            cap.release()
        
        print("Multi-camera tracking system stopped")
    
    def _camera_capture_loop(self, camera_id: int):
        """Camera capture loop for a specific camera"""
        cap = self.video_captures[camera_id]
        while self.running:
            ret, frame = cap.read()
            print(f"[DEBUG] Camera {camera_id} read frame: {ret}")
            if not ret:
                continue
            # Add timestamp
            timestamp = time.time()
            # Put frame in queue (non-blocking)
            try:
                self.frame_queues[camera_id].put_nowait((frame, timestamp))
            except queue.Full:
                # Remove oldest frame and add new one
                try:
                    self.frame_queues[camera_id].get_nowait()
                    self.frame_queues[camera_id].put_nowait((frame, timestamp))
                except queue.Empty:
                    pass
    
    def _processing_loop(self, camera_id: int):
        """Processing loop for a specific camera"""
        detection_save_counter = 0
        save_limit = 500
        cam_folder = f"detections/cam{camera_id+1}"
        os.makedirs(cam_folder, exist_ok=True)
        print(f"[DEBUG] Processing loop started for camera {camera_id}")
        # Helper for crop normalization
        def preprocess_crop(crop):
            # Resize to 128x256 (W x H), convert to RGB, normalize to [0,255]
            crop_resized = cv2.resize(crop, (128, 256))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            return crop_rgb
        while self.running:
            try:
                # Get frame from queue
                frame, timestamp = self.frame_queues[camera_id].get(timeout=0.1)
            except queue.Empty:
                continue
            print(f"[DEBUG] Processing frame for camera {camera_id}")
            # Step 1: Detection
            detections = self.detector.detect(frame)
            # Save first 500 detection frames with bounding boxes and global IDs
            if detection_save_counter < save_limit and len(detections) > 0:
                frame_with_boxes = frame.copy()
                for det in detections:
                    bbox = det['bbox']
                    conf = det['confidence']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Try to get global ID for this detection's track_id
                    label = f"Person {conf:.2f}"
                    track_id = det.get('track_id')
                    global_id = None
                    if track_id is not None and hasattr(self, 'cross_camera_association'):
                        cam_tracks = getattr(self.cross_camera_association, 'camera_tracks', None)
                        if cam_tracks and camera_id in cam_tracks:
                            global_id = cam_tracks[camera_id].get(track_id)
                    if global_id is not None:
                        label += f" (G{global_id})"
                    cv2.putText(frame_with_boxes, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                out_path = os.path.join(cam_folder, f"frame{detection_save_counter+1:04d}.jpg")
                cv2.imwrite(out_path, frame_with_boxes)
                print(f"[DEBUG] Saved detection frame {detection_save_counter+1} for camera {camera_id} at {out_path}")
                detection_save_counter += 1
            # Step 2: Feature extraction (check crops, preprocess)
            person_crops = [preprocess_crop(det['crop']) for det in detections if det.get('crop') is not None and det['crop'].size > 0]
            features = self.feature_extractor.extract_features(person_crops) if person_crops else []
            # Step 3: Single-camera tracking
            feat_idx = 0
            for det in detections:
                if det.get('crop') is not None and det['crop'].size > 0 and feat_idx < len(features):
                    det['features'] = features[feat_idx]
                    feat_idx += 1
                else:
                    det['features'] = None
            tracklets = self.single_trackers[camera_id].update(detections, [det['features'] for det in detections])
            # Prepare result
            result = {
                'camera_id': camera_id,
                'timestamp': timestamp,
                'frame': frame,
                'detections': detections,
                'tracklets': tracklets,
                'fps': self._get_fps(camera_id)
            }
            # Put result in queue (blocking if full)
            while True:
                try:
                    self.result_queues[camera_id].put(result, timeout=0.5)
                    break
                except queue.Full:
                    continue
            # Update FPS counter
            self._update_fps_counter(camera_id)
    
    def _process_frame(self, camera_id: int, frame: np.ndarray, timestamp: float) -> Dict:
        """Process a single frame"""
        # Step 1: Detection
        detections = self.detector.detect(frame)
        
        # Step 2: Feature extraction
        person_crops = [det['crop'] for det in detections]
        features = self.feature_extractor.extract_features(person_crops)
        
        # Step 3: Single-camera tracking
        tracklets = self.single_trackers[camera_id].update(detections, features)
        
        # Prepare result
        result = {
            'camera_id': camera_id,
            'timestamp': timestamp,
            'frame': frame,
            'detections': detections,
            'tracklets': tracklets,
            'fps': self._get_fps(camera_id)
        }
        
        return result
    
    def _association_loop(self):
        """Cross-camera association loop"""
        global_save_counter = 0
        global_save_limit = 500
        os.makedirs('detections/global', exist_ok=True)
        # Open CSV log for global ID assignments
        csv_log_path = 'detections/global/global_id_log.csv'
        csv_log_file = open(csv_log_path, 'w', newline='')
        csv_writer = csv.writer(csv_log_file)
        csv_writer.writerow(['frame_idx', 'camera_id', 'global_id', 'bbox', 'age'])
        frame_idx = 0
        # Color map for global IDs
        def get_color_for_global_id(global_id):
            # Use a fixed palette of distinct colors, cycle if needed
            palette = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
                (0, 128, 255), (128, 255, 0), (255, 128, 0), (0, 255, 128),
                (128, 0, 255), (0, 128, 128), (128, 128, 0), (128, 128, 128)
            ]
            return palette[global_id % len(palette)]
        # For duplicate/jumping ID detection
        prev_global_assignments = {0: {}, 1: {}}  # camera_id: {track_id: global_id}
        duplicate_log_path = 'detections/global/duplicate_id_log.csv'
        duplicate_log_file = open(duplicate_log_path, 'w', newline='')
        duplicate_writer = csv.writer(duplicate_log_file)
        duplicate_writer.writerow(['frame_idx', 'camera_id', 'global_id', 'event', 'bbox'])
        while self.running:
            # Collect tracklets from all cameras
            camera_tracklets = {}
            for camera_id in self.video_captures:
                try:
                    result = self.result_queues[camera_id].get(timeout=0.1)
                    camera_tracklets[camera_id] = result['tracklets']
                    self.latest_results[camera_id] = result
                except queue.Empty:
                    continue
            # Perform cross-camera association
            if camera_tracklets:
                for camera_id, tracklets in camera_tracklets.items():
                    # Only update with confirmed tracks (hits >= 3, state == 'confirmed')
                    confirmed_tracklets = [t for t in tracklets if t.get('hits', 0) >= 3 and t.get('age', 0) >= 5 and t.get('time_since_update', 0) == 0]
                    self.cross_camera_association.update_camera_tracks(camera_id, confirmed_tracklets)
                    # Debug print: show local to global ID mapping
                    if hasattr(self.cross_camera_association, 'camera_tracks'):
                        print(f"[DEBUG] Camera {camera_id} local->global: {self.cross_camera_association.camera_tracks[camera_id]}")
                # Update global tracks
                self.global_tracks = self.cross_camera_association.get_global_tracks()
                # Debug print for global ID creation/assignment
                for global_id, track in self.global_tracks.items():
                    print(f"[DEBUG] Global ID {global_id} active in cameras: {track['cameras']}")
            # Save side-by-side global detection image if both cameras have results
            if global_save_counter < global_save_limit and 0 in self.latest_results and 1 in self.latest_results:
                frame1 = self.latest_results[0]['frame'].copy()
                frame2 = self.latest_results[1]['frame'].copy()
                # Draw only global tracks (filter out short-lived tracks: age < 5)
                global_tracks = self.get_global_tracks()
                def draw_global_tracks(frame, camera_id, global_tracks):
                    # For duplicate/jumping ID detection in this frame
                    global_id_to_bboxes = {}
                    track_id_to_global = {}
                    for global_id, track in global_tracks.items():
                        bboxes = track.get('bboxes', {})
                        age = 0
                        if camera_id in bboxes and 'track_length' in track:
                            age = track['track_length']
                        if camera_id in bboxes and age >= 5:
                            bbox = bboxes[camera_id]
                            x1, y1, x2, y2 = map(int, bbox)
                            color = get_color_for_global_id(global_id)
                            label = f"G{global_id}"
                            # Check for duplicate global IDs (same global_id assigned to multiple bboxes in this camera)
                            if global_id not in global_id_to_bboxes:
                                global_id_to_bboxes[global_id] = []
                            global_id_to_bboxes[global_id].append(bbox)
                            # Check for jumping IDs (track_id changes global_id between frames)
                            # We use the center of the bbox as a proxy for track_id (since only global tracks are visualized)
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            track_id_proxy = (cx, cy)
                            track_id_to_global[track_id_proxy] = global_id
                            # Draw normal or highlight if duplicate/jumping
                            highlight = False
                            # Highlight duplicate
                            if len(global_id_to_bboxes[global_id]) > 1:
                                highlight = True
                                duplicate_writer.writerow([global_save_counter+1, camera_id, global_id, 'duplicate', bbox])
                                print(f"[WARNING] Duplicate global ID {global_id} in camera {camera_id} at frame {global_save_counter+1}")
                            # Highlight jumping
                            prev_global = prev_global_assignments[camera_id].get(track_id_proxy)
                            if prev_global is not None and prev_global != global_id:
                                highlight = True
                                duplicate_writer.writerow([global_save_counter+1, camera_id, global_id, 'jumping', bbox])
                                print(f"[WARNING] Jumping global ID for track {track_id_proxy} in camera {camera_id} at frame {global_save_counter+1}: {prev_global} -> {global_id}")
                            # Draw bbox
                            box_color = (0, 0, 255) if highlight else color
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                            # Draw trajectory (last 20 positions)
                            if 'camera_tracklets' in track:
                                tracklets = track['camera_tracklets'].get(camera_id, [])
                                points = []
                                for t in tracklets[-20:]:
                                    bx = t['bbox']
                                    cx = int((bx[0] + bx[2]) / 2)
                                    cy = int((bx[1] + bx[3]) / 2)
                                    points.append((cx, cy))
                                if len(points) > 1:
                                    cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color, 2)
                            # Log to CSV
                            csv_writer.writerow([global_save_counter+1, camera_id, global_id, bbox, age])
                    # Update prev_global_assignments for next frame
                    prev_global_assignments[camera_id] = track_id_to_global
                draw_global_tracks(frame1, 0, global_tracks)
                draw_global_tracks(frame2, 1, global_tracks)
                # Resize frames to same height
                h = max(frame1.shape[0], frame2.shape[0])
                w1 = int(frame1.shape[1] * h / frame1.shape[0])
                w2 = int(frame2.shape[1] * h / frame2.shape[0])
                frame1_resized = cv2.resize(frame1, (w1, h))
                frame2_resized = cv2.resize(frame2, (w2, h))
                combined = cv2.hconcat([frame1_resized, frame2_resized])
                out_path = f"detections/global/global_frame_{global_save_counter+1:04d}.jpg"
                cv2.imwrite(out_path, combined)
                print(f"[DEBUG] Saved global detection frame {global_save_counter+1} at {out_path}")
                global_save_counter += 1
            frame_idx += 1
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
        csv_log_file.close()
        duplicate_log_file.close()
    
    def _update_fps_counter(self, camera_id: int):
        """Update FPS counter for a camera"""
        counter = self.fps_counters[camera_id]
        counter['count'] += 1
        
        # Reset counter every second
        current_time = time.time()
        if current_time - counter['start_time'] >= 1.0:
            counter['fps'] = counter['count']
            counter['count'] = 0
            counter['start_time'] = current_time
    
    def _get_fps(self, camera_id: int) -> float:
        """Get current FPS for a camera"""
        return self.fps_counters[camera_id].get('fps', 0)
    
    def get_latest_results(self) -> Dict:
        """Get latest results from all cameras"""
        return self.latest_results.copy()
    
    def get_global_tracks(self) -> Dict:
        """Get current global tracks"""
        return self.global_tracks.copy()
    
    def visualize_results(self, display_size=(1920, 1080)):
        """Visualize tracking results in separate windows for each camera, with interactive controls"""
        window_names = {}
        for i, config in enumerate(self.camera_configs):
            name = config.get('name', f'Camera {i}')
            window_names[i] = name
        print(f"[INFO] Visualization windows: {window_names}")
        paused = False
        step = False
        while self.running:
            if paused and not step:
                key = cv2.waitKey(0)
                if key == 32:  # Spacebar to resume
                    paused = False
                elif key == 27:  # ESC to exit
                    break
                elif key == 83 or key == 2555904:  # Right arrow to step
                    step = True
                continue
            results = self.get_latest_results()
            if not results:
                time.sleep(0.1)
                continue
            for i in self.active_cameras:
                if i in results:
                    result = results[i]
                    frame = result['frame']
                    # Resize frame to display size
                    frame_resized = cv2.resize(frame, tuple(display_size))
                    # Draw tracklets
                    frame_with_tracks = self._draw_tracklets(
                        frame_resized, result['tracklets'], i
                    )
                    # Add FPS and camera info
                    fps = result.get('fps', 0)
                    cv2.putText(frame_with_tracks, f"{window_names[i]} - FPS: {fps:.1f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if paused:
                        cv2.putText(frame_with_tracks, "[PAUSED]", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.imshow(window_names[i], frame_with_tracks)
                else:
                    # If no result, show a black frame with message
                    panel = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
                    cv2.putText(panel, f"{window_names[i]} not available", (30, display_size[1] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                    if paused:
                        cv2.putText(panel, "[PAUSED]", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.imshow(window_names[i], panel)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
            elif key == 32:  # Spacebar to pause
                paused = True
                step = False
            elif key == 83 or key == 2555904:  # Right arrow to step
                paused = True
                step = True
            else:
                step = False
        # Close all windows
        for name in window_names.values():
            cv2.destroyWindow(name)
        cv2.destroyAllWindows()
    
    def _draw_tracklets(self, frame: np.ndarray, tracklets: List[Dict], camera_id: int) -> np.ndarray:
        """Draw tracklets on frame"""
        frame_with_tracks = frame.copy()
        
        # Color map for different tracks
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for tracklet in tracklets:
            track_id = tracklet['track_id']
            bbox = tracklet['bbox']
            
            # Get global ID if available
            global_id = None
            if camera_id in self.cross_camera_association.camera_tracks:
                global_id = self.cross_camera_association.camera_tracks[camera_id].get(track_id)
            
            # Choose color based on track ID
            color = colors[track_id % len(colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_with_tracks, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"T{track_id}"
            if global_id is not None:
                label += f" (G{global_id})"
            
            cv2.putText(frame_with_tracks, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame_with_tracks
    
    def save_results(self, output_path: str):
        """Save tracking results to file"""
        # Implementation for saving results
        pass
    
    def get_system_stats(self) -> Dict:
        """Get system performance statistics"""
        stats = {
            'num_cameras': self.num_cameras,
            'global_tracks': len(self.global_tracks),
            'camera_fps': {cam_id: self._get_fps(cam_id) for cam_id in self.video_captures},
            'total_tracklets': sum(len(self.single_trackers[cam_id].tracks) 
                                 for cam_id in self.single_trackers),
            'running': self.running
        }
        return stats

# Example usage
if __name__ == "__main__":
    # Configure cameras
    camera_configs = [
        {
            'source': r"D:\ANITUM\cam1.mp4",
            'width': 640,
            'height': 480,
            'fps': 30
        },
        {
            'source': r"D:\ANITUM\cam5.mp4", 
            'width': 640,
            'height': 480,
            'fps': 30
        }
    ]
    
    # Initialize system
    system = MultiCameraTrackingSystem(camera_configs, feature_extractor_type='resnet50')
    
    try:
        # Initialize cameras
        system.initialize_cameras()
        
        # Start tracking
        system.start_tracking()
        
        # Start visualization
        system.visualize_results()
        
    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        system.stop_tracking()
        print("System stopped")

# For best results, tune your config.json as follows:
# "detection": {"confidence_threshold": 0.3},
# "tracking": {"max_disappeared": 80, "max_distance": 120},
# "association": {"similarity_threshold": 0.55, "time_window": 800, "max_gallery_size": 2000}