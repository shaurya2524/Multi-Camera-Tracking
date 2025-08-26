#!/usr/bin/env python3
"""
Enhanced Multi-camera tracking pipeline with Global ID management - FIXED VERSION
Usage: python enhanced_tracking_pipeline.py video1.mp4 video2.mp4 [--mode live|batch]
"""

import sys
import os
import cv2
import numpy as np
import argparse
from yolo_detector import YOLOv8Detector
from yolo_multicam_tracker import YoloMultiCameraTracker
from multi_camera_tracker import GlobalIDManager


class EnhancedTrackingPipeline:
    def __init__(self):
        # Original components
        self.detector = YOLOv8Detector()
        self.tracker = YoloMultiCameraTracker()
        self.global_id_manager = GlobalIDManager(
            distance_threshold=50.0,
            confidence_threshold=0.6,
            max_missing_frames=50,
            min_track_length=5,
            merge_iou_threshold=0.3
        )

        self.recording = False
        self.output_writer = None
        self.combined_writer = None   # <-- FIX: initialize combined writer
        self.frame_count = 0
        self.last_detections1 = []
        self.last_detections2 = []
        self._setup_homography()

        
    def _setup_homography(self):
        """Setup homography matrix for global coordinate transformation"""
        try:
            homography = np.load('homography_matrix.npy')
            self.global_id_manager.set_homography(1, homography)
            print("Homography matrix loaded successfully")
        except FileNotFoundError:
            print("Warning: homography_matrix.npy not found. Global tracking accuracy may be reduced.")
            
    def extract_tracker_detections(self, camera_id):
        """SIMPLE DIRECT FIX - Just use YOLO detections with frame-based IDs"""
        detections = []
        
        # Get the YOLO detections for this camera
        if camera_id == 1:
            yolo_detections = self.last_detections1
        elif camera_id == 2:
            yolo_detections = self.last_detections2
        else:
            return detections
        
        print(f"Processing camera {camera_id}: {type(yolo_detections)}, {len(yolo_detections) if yolo_detections else 0} detections")
        
        # Handle different detection formats
        if yolo_detections is None:
            return detections
            
        # Check if it's a list/array of detections
        if hasattr(yolo_detections, '__len__') and len(yolo_detections) > 0:
            for i, det in enumerate(yolo_detections):
                try:
                    if isinstance(det, dict):
                        bbox = det.get("bbox", (0, 0, 0, 0))
                        conf = float(det.get("conf", 0.8))
                        class_id = det.get("cls", 0)
                        local_id = f"c{camera_id}_f{self.frame_count}_d{i}"
                        detections.append((local_id, bbox, conf))
                        print(f"  Added dict detection: {local_id}, bbox={bbox}, conf={conf}")
                    # Handle different detection formats
                    if isinstance(det, (list, tuple)) and len(det) >= 4:
                        # Format: [x1, y1, x2, y2, conf, class_id]
                        if len(det) >= 6:
                            x1, y1, x2, y2, conf, class_id = det[:6]
                        elif len(det) >= 5:
                            x1, y1, x2, y2, conf = det[:5]
                            class_id = 0
                        else:
                            x1, y1, x2, y2 = det[:4]
                            conf = 0.8
                            class_id = 0
                        
                        # Convert to (x, y, w, h) format
                        bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                        # Create a simple ID based on frame and detection index
                        local_id = f"c{camera_id}_f{self.frame_count}_d{i}"
                        detections.append((local_id, bbox, float(conf)))
                        print(f"  Added detection: {local_id}, bbox={bbox}, conf={conf}")
                        
                    # Handle numpy array format
                    elif hasattr(det, 'shape') and len(det) >= 4:
                        x1, y1, x2, y2 = det[:4]
                        conf = det[4] if len(det) > 4 else 0.8
                        bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                        local_id = f"c{camera_id}_f{self.frame_count}_d{i}"
                        detections.append((local_id, bbox, float(conf)))
                        print(f"  Added detection: {local_id}, bbox={bbox}, conf={conf}")
                        
                except Exception as e:
                    print(f"  Error processing detection {i}: {e}")
                    continue
        
        print(f"Camera {camera_id} final detections: {len(detections)}")
        return detections
    def is_near_boundary(self, bbox, frame, margin=50):
        """Check if a bounding box is close to the border of the frame"""
        x, y, w, h = map(int, bbox)  # ensure scalars
        H, W = frame.shape[:2]
        return (x < margin) or (y < margin) or (x + w > W - margin) or (y + h > H - margin)


    
    def setup_recording(self, output_path, fps, width, height):
        """Setup video recording"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.recording = True
        print(f"Recording to {output_path}")
    
    def process_frame(self, frame1, frame2):
        """Enhanced frame processing with global ID management and boundary-based motion tracking"""
        self.frame_count += 1
        print(f"\n--- Processing Frame {self.frame_count} ---")

        # 1. Run YOLO detections
        detections1 = self.detector.detect(frame1)
        detections2 = self.detector.detect(frame2)

        # Store detections for fallback extraction
        self.last_detections1 = detections1
        self.last_detections2 = detections2

        print(f"YOLO Detections - Cam1: {len(detections1)}, Cam2: {len(detections2)}")

        # 2. Update single-camera tracker
        self.tracker.update(detections1, detections2)
        print("Tracker updated")

        # 3. Extract detections for global ID manager
        camera_detections = {
            1: self.extract_tracker_detections(1),
            2: self.extract_tracker_detections(2)
        }
        print(f"Extracted for Global ID - Cam1: {len(camera_detections[1])}, Cam2: {len(camera_detections[2])}")

        # 4. Process with global ID manager (initial pass)
        global_tracks = self.global_id_manager.process_detections(camera_detections)
        print(f"Global tracks after processing: {len(global_tracks)}")

        # 5. Create union view (needed for boundary checks)
        union_frame = self.create_union_frame(frame1, frame2)

        # 6. Boundary-based re-ID (motion tracker correction)
        for cam_id, dets in camera_detections.items():
            for det in dets:
                local_id, bbox, conf = det
                if self.is_near_boundary(bbox, union_frame):
                    matched_id = self.global_id_manager.match_with_predictions(bbox)
                    if matched_id:
                        print(f"[Boundary Motion Tracker] Re-using Global ID {matched_id}")
                        self.global_id_manager.assign_existing_id(
                            matched_id,
                            self.global_id_manager.create_detection(cam_id, local_id, bbox, conf),
                            cam_id
                        )

        # 7. Draw tracker overlays
        self.tracker.draw_tracks(frame1, 1)
        self.tracker.draw_tracks(frame2, 2)
        self.tracker.draw_tracks(None, None, union_frame)

        # 8. Draw global overlays
        self.draw_global_id_overlay(frame1, 1, global_tracks)
        self.draw_global_id_overlay(frame2, 2, global_tracks)
        self.draw_global_tracks_union(union_frame, global_tracks)

        # 9. Add info overlays
        self.add_enhanced_info_overlay(union_frame, global_tracks)

        return frame1, frame2, union_frame

    
    def create_union_frame(self, frame1, frame2):
        """Create union frame from two camera views"""
        h2, w2 = frame2.shape[:2]
        union_w, union_h = w2 + 200, h2 + 200
        union_frame = np.zeros((union_h, union_w, 3), dtype=np.uint8)
        
        # Place frame2 (reference)
        union_frame[:h2, :w2] = frame2
        
        # Warp and blend frame1
        try:
            warped1 = cv2.warpPerspective(frame1, np.load('homography_matrix.npy'), (union_w, union_h))
            mask1_warped = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY)
            mask1_warped = (mask1_warped > 0).astype(np.uint8) * 255
            
            for c in range(3):
                union_frame[:, :, c] = np.where(mask1_warped > 0, 
                                               (union_frame[:, :, c] * 0.5 + warped1[:, :, c] * 0.5).astype(np.uint8),
                                               union_frame[:, :, c])
        except FileNotFoundError:
            # Fallback: simple placement if homography not available
            h1, w1 = frame1.shape[:2]
            if h1 <= union_h and w1 <= union_w - w2:
                union_frame[:h1, w2:w2+w1] = frame1
        
        return union_frame
    
    def draw_global_id_overlay(self, frame, camera_id, global_tracks):
        """Draw global IDs as overlay on existing tracks"""
        active_tracks = self.global_id_manager.get_active_global_tracks()
        
        for global_id, track in active_tracks.items():
            if camera_id in track.camera_detections:
                detection = track.camera_detections[camera_id]
                x, y, w, h = detection.bbox
                
                # Draw global ID badge
                badge_x, badge_y = x + w - 30, y - 5
                cv2.rectangle(frame, (badge_x, badge_y), (badge_x + 25, badge_y + 15), (0, 255, 255), -1)
                cv2.putText(frame, f"G{global_id}", (badge_x + 2, badge_y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    def draw_global_tracks_union(self, union_frame, global_tracks):
        """Draw global tracks on union view"""
        active_tracks = self.global_id_manager.get_active_global_tracks()
        
        for global_id, track in active_tracks.items():
            if track.position_history:
                center = track.position_history[-1]
                center_int = (int(center[0]), int(center[1]))
                
                # Color based on number of cameras tracking
                if len(track.active_cameras) > 1:
                    color = (0, 255, 255)  # Yellow for multi-camera
                    thickness = 3
                else:
                    color = self._get_track_color(global_id)
                    thickness = 2
                
                # Draw global track point
                cv2.circle(union_frame, center_int, 8, color, thickness)
                
                # Draw global ID
                cv2.putText(union_frame, f"G{global_id}", 
                           (center_int[0] - 15, center_int[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw track history trail
                if len(track.position_history) > 1:
                    points = [(int(p[0]), int(p[1])) for p in track.position_history[-10:]]
                    for i in range(1, len(points)):
                        alpha = i / len(points)  # Fade effect
                        line_color = tuple(int(c * alpha) for c in color)
                        cv2.line(union_frame, points[i-1], points[i], line_color, 2)
    
    def _get_track_color(self, global_id):
        """Generate consistent color for global track"""
        np.random.seed(global_id)
        return tuple(np.random.randint(50, 255, 3).tolist())
    
    def add_enhanced_info_overlay(self, frame, global_tracks):
        """Enhanced information overlay with global tracking stats"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Get statistics
        stats = self.global_id_manager.get_statistics()
        active_tracks = len([t for t in global_tracks.values() if len(t.active_cameras) > 0])
        multi_cam_tracks = len([t for t in global_tracks.values() if len(t.active_cameras) > 1])
        
        # Enhanced text information
        y_offset = 30
        cv2.putText(frame, "Enhanced YOLOv8 Multi-Camera Tracking", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Frame: {self.frame_count} | Active Global Tracks: {active_tracks}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        y_offset += 20
        cv2.putText(frame, f"Multi-Camera Tracks: {multi_cam_tracks}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        y_offset += 20
        # cv2.putText(frame, f"Cam1: {stats['tracks_per_camera'].get('camera_1', 0)} | Cam2: {stats['tracks_per_camera'].get('camera_2', 0)}", 
                #    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        y_offset += 20
        cv2.putText(frame, f"Total Global IDs Generated: {stats['next_global_id'] - 1}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 70, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def debug_tracker_state(self):
        """Debug function to inspect tracker state"""
        print("\n--- TRACKER DEBUG INFO ---")
        print(f"Tracker type: {type(self.tracker)}")
        print(f"Tracker attributes: {dir(self.tracker)}")
        
        if hasattr(self.tracker, 'tracks'):
            tracks = self.tracker.tracks
            print(f"Tracks type: {type(tracks)}")
            print(f"Tracks length: {len(tracks) if tracks else 0}")
            
            if tracks:
                first_track = next(iter(tracks.values())) if isinstance(tracks, dict) else tracks[0]
                print(f"First track type: {type(first_track)}")
                print(f"First track attributes: {dir(first_track)}")
        print("--- END DEBUG INFO ---\n")
    
    def run_live_tracking(self, video1_path, video2_path):
        """Run enhanced live tracking with global IDs"""
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        # Get video properties
        fps = cap1.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        
        print("\n=== Enhanced YOLOv8 Multi-Camera Tracking with Global IDs ===")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  R - Start/Stop recording")
        print("  S - Show global tracking statistics")
        print("  C - Clear old global tracks")
        print("  D - Debug tracker state")
        print("  Q - Quit")
        print("-" * 60)
        
        paused = False
        
        while True:
            if not paused:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    # Loop videos
                    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Process frames with global ID tracking
                tracked_frame1, tracked_frame2, union_frame = self.process_frame(frame1, frame2)

                # Concatenate tracked_frame1 and tracked_frame2 side by side
                combined_frame = cv2.hconcat([tracked_frame1, tracked_frame2])
                
                # Record if enabled
                if self.recording:
                    if self.output_writer:
                        self.output_writer.write(union_frame)
                    if self.combined_writer:
                        self.combined_writer.write(combined_frame)
            
            # Display frames
            cv2.imshow('Camera 1 - Enhanced Tracking', tracked_frame1)
            cv2.imshow('Camera 2 - Enhanced Tracking', tracked_frame2)
            cv2.imshow('Union - Global ID Tracking', union_frame)
            cv2.imshow('Combined Side-by-Side', combined_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):
                if not self.recording:
                    h_union, w_union = union_frame.shape[:2]
                    self.setup_recording('enhanced_tracking_output.mp4', fps, w_union, h_union)

                    # Use actual combined_frame dimensions
                    combined_frame = cv2.hconcat([tracked_frame1, tracked_frame2])
                    h_comb, w_comb = combined_frame.shape[:2]
                    self.combined_writer = cv2.VideoWriter(
                        'combined_output.mp4',
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (w_comb, h_comb)
                    )

                    self.recording = True
                    print("Recording started")
                else:
                    self.recording = False
                    if self.output_writer:
                        self.output_writer.release()
                        self.output_writer = None
                    if self.combined_writer:
                        self.combined_writer.release()
                        self.combined_writer = None
                    print("Recording stopped")


            elif key == ord('s'):
                self.print_tracking_statistics()
            elif key == ord('c'):
                old_count = len(self.global_id_manager.global_tracks)
                self.global_id_manager._cleanup_old_tracks()
                new_count = len(self.global_id_manager.global_tracks)
                print(f"Cleared {old_count - new_count} old tracks")
            elif key == ord('d'):
                self.debug_tracker_state()

        
        # Cleanup
        cap1.release()
        cap2.release()
        if self.output_writer:
            self.output_writer.release()
        if self.combined_writer:
            self.combined_writer.release()
        cv2.destroyAllWindows()

        
        # Print final statistics
        print("\n=== Final Tracking Session Summary ===")
        self.print_tracking_statistics()
    
    def print_tracking_statistics(self):
        """Print detailed tracking statistics"""
        stats = self.global_id_manager.get_statistics()
        active_tracks = self.global_id_manager.get_active_global_tracks()
        
        print("\n--- Global Tracking Statistics ---")
        print(f"Total Frames Processed: {stats['frame_count']}")
        print(f"Total Global IDs Created: {stats['next_global_id'] - 1}")
        print(f"Currently Active Tracks: {stats['active_global_tracks']}")
        print(f"Camera 1 Active Tracks: {stats['tracks_per_camera'].get('camera_1', 0)}")
        print(f"Camera 2 Active Tracks: {stats['tracks_per_camera'].get('camera_2', 0)}")
        
        # Multi-camera track analysis
        multi_cam_tracks = [t for t in active_tracks.values() if len(t.active_cameras) > 1]
        print(f"Multi-Camera Tracks: {len(multi_cam_tracks)}")
        
        if multi_cam_tracks:
            avg_track_length = sum(len(t.position_history) for t in multi_cam_tracks) / len(multi_cam_tracks)
            print(f"Average Multi-Camera Track Length: {avg_track_length:.1f} frames")
        
        print("-" * 35)
    
    def run_batch_processing(self, video1_path, video2_path, output_path):
        """Run batch processing with global ID tracking"""
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        # Get video properties
        fps = cap1.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get frame dimensions
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Error reading video files")
            return
        
        union_frame = self.create_union_frame(frame1, frame2)
        h, w = union_frame.shape[:2]
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Reset videos
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print(f"Processing {total_frames} frames with global ID tracking...")
        
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Process frames with enhanced tracking
            _, _, union_frame = self.process_frame(frame1, frame2)
            
            # Write frame
            out.write(union_frame)
            
            # Progress indicator
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")
        
        # Cleanup
        cap1.release()
        cap2.release()
        out.release()
        
        print(f"Enhanced batch processing complete. Output saved to {output_path}")
        self.print_tracking_statistics()


def main():
    parser = argparse.ArgumentParser(description='Enhanced YOLOv8 Multi-camera tracking with Global IDs')
    parser.add_argument('video1', help='Path to first video')
    parser.add_argument('video2', help='Path to second video')
    parser.add_argument('--mode', choices=['live', 'batch'], default='live',
                       help='Processing mode (live=interactive, batch=process to file)')
    parser.add_argument('--output', default='enhanced_tracked_output.mp4',
                       help='Output video path for batch mode')
    parser.add_argument('--distance-threshold', type=float, default=50.0,
                       help='Distance threshold for global ID matching (pixels)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Minimum confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Check if videos exist
    if not os.path.exists(args.video1) or not os.path.exists(args.video2):
        print("Error: Video files not found")
        sys.exit(1)
    
    # Check if homography matrix exists (optional but recommended)
    if not os.path.exists('homography_matrix.npy'):
        print("Warning: homography_matrix.npy not found")
        print("Global tracking will work but may be less accurate")
        print("Consider running your camera calibration first")
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create enhanced pipeline
    pipeline = EnhancedTrackingPipeline()
    
    # Configure global ID manager with command line parameters
    pipeline.global_id_manager.distance_threshold = args.distance_threshold
    pipeline.global_id_manager.confidence_threshold = args.confidence_threshold
    
    # Run pipeline
    if args.mode == 'live':
        pipeline.run_live_tracking(args.video1, args.video2)
    else:
        pipeline.run_batch_processing(args.video1, args.video2, args.output)


if __name__ == "__main__":
    main()