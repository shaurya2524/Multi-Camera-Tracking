#!/usr/bin/env python3
"""
Multi-camera tracking pipeline that integrates with your existing view union system
Usage: python tracking_pipeline.py video1.mp4 video2.mp4 [--method simple|kalman]
"""

import sys
import os
import cv2
import numpy as np
import argparse
from yolo_detector import YOLOv8Detector
from yolo_multicam_tracker import YoloMultiCameraTracker

class TrackingPipeline:
    def __init__(self):
        self.detector = YOLOv8Detector()
        self.tracker = YoloMultiCameraTracker()
        self.recording = False
        self.output_writer = None
        self.frame_count = 0
        
    def setup_recording(self, output_path, fps, width, height):
        """Setup video recording"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.recording = True
        print(f"Recording to {output_path}")
    
    def process_frame(self, frame1, frame2):
        """Process a single frame pair"""
        self.frame_count += 1
        detections1 = self.detector.detect(frame1)
        detections2 = self.detector.detect(frame2)
        self.tracker.update(detections1, detections2)
        union_frame = self.create_union_frame(frame1, frame2)
        self.tracker.draw_tracks(frame1, 1)
        self.tracker.draw_tracks(frame2, 2)
        self.tracker.draw_tracks(None, None, union_frame)
        self.add_info_overlay(union_frame)
        return frame1, frame2, union_frame
    
    def create_union_frame(self, frame1, frame2):
        """Create union frame from two camera views"""
        h2, w2 = frame2.shape[:2]
        union_w, union_h = w2 + 200, h2 + 200
        union_frame = np.zeros((union_h, union_w, 3), dtype=np.uint8)
        
        # Place frame2 (reference)
        union_frame[:h2, :w2] = frame2
        
        # Warp and blend frame1
        warped1 = cv2.warpPerspective(frame1, np.load('homography_matrix.npy'), (union_w, union_h))
        mask1_warped = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY)
        mask1_warped = (mask1_warped > 0).astype(np.uint8) * 255
        
        for c in range(3):
            union_frame[:, :, c] = np.where(mask1_warped > 0, 
                                           (union_frame[:, :, c] * 0.5 + warped1[:, :, c] * 0.5).astype(np.uint8),
                                           union_frame[:, :, c])
        
        return union_frame
    
    def add_info_overlay(self, frame):
        """Add tracking information overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text info
        y_offset = 35
        cv2.putText(frame, f"YOLOv8 Tracking", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 70, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def run_live_tracking(self, video1_path, video2_path):
        """Run live tracking with interactive controls"""
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        # Get video properties
        fps = cap1.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        
        print("\n=== YOLOv8 Multi-Camera Tracking ===")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  R - Start/Stop recording")
        print("  Q - Quit")
        print("-" * 40)
        
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
                
                # Process frames
                tracked_frame1, tracked_frame2, union_frame = self.process_frame(frame1, frame2)
                
                # Record if enabled
                if self.recording and self.output_writer:
                    self.output_writer.write(union_frame)
            
            # Display frames
            cv2.imshow('Camera 1', tracked_frame1)
            cv2.imshow('Camera 2', tracked_frame2)
            cv2.imshow('Union - YOLOv8 Tracking', union_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):
                if not self.recording:
                    h, w = union_frame.shape[:2]
                    self.setup_recording('tracking_output.mp4', fps, w, h)
                else:
                    self.recording = False
                    if self.output_writer:
                        self.output_writer.release()
                    print("Recording stopped")
        
        # Cleanup
        cap1.release()
        cap2.release()
        if self.output_writer:
            self.output_writer.release()
        cv2.destroyAllWindows()
    
    def run_batch_processing(self, video1_path, video2_path, output_path):
        """Run batch processing to create tracked video"""
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
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Process frames
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
        
        print(f"Batch processing complete. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Multi-camera tracking pipeline')
    parser.add_argument('video1', help='Path to first video')
    parser.add_argument('video2', help='Path to second video')
    parser.add_argument('--mode', choices=['live', 'batch'], default='live',
                       help='Processing mode')
    parser.add_argument('--output', default='tracked_output.mp4',
                       help='Output video path for batch mode')
    args = parser.parse_args()
    
    # Check if videos exist
    if not os.path.exists(args.video1) or not os.path.exists(args.video2):
        print("Error: Video files not found")
        sys.exit(1)
    
    # Check if homography matrix exists
    if not os.path.exists('homography_matrix.npy'):
        print("Error: homography_matrix.npy not found")
        print("Please run your existing pipeline first to generate the homography matrix")
        sys.exit(1)
    
    # Create pipeline
    pipeline = TrackingPipeline()
    
    # Run pipeline
    if args.mode == 'live':
        pipeline.run_live_tracking(args.video1, args.video2)
    else:
        pipeline.run_batch_processing(args.video1, args.video2, args.output)

if __name__ == "__main__":
    main()