import cv2
import numpy as np
from shapely.geometry import Polygon
import json
import time

class ViewUnion:
    def __init__(self):
        self.homography = None
        
    def load_homography(self, file_path='homography_matrix.npy'):
        self.homography = np.load(file_path)
        
    def get_overlap_area(self, frame1_shape, frame2_shape):
        h1, w1 = frame1_shape[:2]
        h2, w2 = frame2_shape[:2]
        
        # Get corners of frame1
        corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
        
        # Transform corners using homography
        warped_corners = cv2.perspectiveTransform(corners1.reshape(-1, 1, 2), self.homography)
        warped_corners = warped_corners.reshape(-1, 2)
        
        # Frame2 corners
        corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        
        # Calculate intersection using Shapely
        poly1 = Polygon(warped_corners)
        poly2 = Polygon(corners2)
        
        intersection = poly1.intersection(poly2)
        
        if intersection.is_empty:
            return None, 0
        
        return intersection, intersection.area
    
    def create_union_video(self, video1_path, video2_path, output_path='union_output.mp4'):
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        # Get video properties
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error reading videos")
            return
        
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Calculate union dimensions
        union_w = max(w1, w2) + 100  # Add margin
        union_h = max(h1, h2) + 100
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (union_w, union_h))
        
        # Get overlap info
        overlap_poly, overlap_area = self.get_overlap_area(frame1.shape, frame2.shape)
        print(f"Overlap area: {overlap_area:.2f} pixels")
        
        # Reset video captures
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Create union frame
            union_frame = np.zeros((union_h, union_w, 3), dtype=np.uint8)
            
            # Place frame2 (reference)
            union_frame[:h2, :w2] = frame2
            
            # Warp and blend frame1
            warped1 = cv2.warpPerspective(frame1, self.homography, (union_w, union_h))
            
            # Simple blending in overlap area
            mask = (warped1 > 0).astype(np.uint8)
            union_frame = cv2.bitwise_or(union_frame, warped1)
            
            # Write frame
            out.write(union_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap1.release()
        cap2.release()
        out.release()
        print(f"Union video saved as {output_path}")
    
    def show_verification_view(self, video1_path, video2_path):
        """Show original videos side by side with union for verification"""
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        # Get video properties
        fps = cap1.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        # Get first frames to determine sizes
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error reading videos")
            return
        
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Calculate display dimensions - make them larger and uniform
        target_height = 300  # Fixed height for all videos
        display_w1 = int(w1 * target_height / h1)
        display_h1 = target_height
        display_w2 = int(w2 * target_height / h2)
        display_h2 = target_height
        
        # Union dimensions - make it more visible
        union_w = max(w1, w2) + 200
        union_h = max(h1, h2) + 200
        display_union_w = int(union_w * target_height / union_h)
        display_union_h = target_height
        
        # Create combined display with proper spacing
        spacing = 10
        combined_width = display_w1 + display_w2 + display_union_w + (3 * spacing)
        combined_height = target_height + 80  # More space for text
        
        # Reset video captures
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        paused = False
        frame_count = 0
        
        print("Verification view controls:")
        print("- Press 'q' to quit")
        print("- Press 'SPACE' to pause/resume")
        print("- Press 'r' to restart videos")
        
        while True:
            start_time = time.time()
            
            if not paused:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    # Loop videos
                    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_count += 1
            
            # Create union frame - fix the blending issue
            union_frame = np.zeros((union_h, union_w, 3), dtype=np.uint8)
            
            # Place frame2 first (reference frame)
            union_frame[:h2, :w2] = frame2
            
            # Warp frame1 and blend properly
            warped1 = cv2.warpPerspective(frame1, self.homography, (union_w, union_h))
            
            # Create mask for warped frame1
            mask1 = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY)
            mask1 = (mask1 > 0).astype(np.uint8) * 255
            
            # Blend the frames where they overlap
            for c in range(3):
                union_frame[:, :, c] = np.where(mask1 > 0, 
                                               (union_frame[:, :, c] * 0.5 + warped1[:, :, c] * 0.5).astype(np.uint8),
                                               union_frame[:, :, c])
            
            # Resize frames for display
            display_frame1 = cv2.resize(frame1, (display_w1, display_h1))
            display_frame2 = cv2.resize(frame2, (display_w2, display_h2))
            display_union = cv2.resize(union_frame, (display_union_w, display_union_h))
            
            # Create combined display with white background
            combined_frame = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 50  # Dark gray background
            
            # Place video1
            y_offset = 50
            combined_frame[y_offset:y_offset+display_h1, spacing:spacing+display_w1] = display_frame1
            cv2.putText(combined_frame, 'Video 1', (spacing + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Place video2
            x_offset = spacing + display_w1 + spacing
            combined_frame[y_offset:y_offset+display_h2, x_offset:x_offset+display_w2] = display_frame2
            cv2.putText(combined_frame, 'Video 2', (x_offset + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Place union
            x_offset = spacing + display_w1 + spacing + display_w2 + spacing
            combined_frame[y_offset:y_offset+display_union_h, x_offset:x_offset+display_union_w] = display_union
            cv2.putText(combined_frame, 'Union', (x_offset + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add frame counter and status
            status_text = f"Frame: {frame_count} | {'PAUSED' if paused else 'PLAYING'} | FPS: {fps:.1f}"
            cv2.putText(combined_frame, status_text, (10, combined_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add control instructions
            controls_text = "Controls: SPACE=pause/resume, R=restart, Q=quit"
            cv2.putText(combined_frame, controls_text, (10, combined_height - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Verification View - Original Videos and Union', combined_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause/resume
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):  # Restart videos
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                paused = False
                print("Restarted videos")
            
            # Maintain proper frame rate
            if not paused:
                elapsed = time.time() - start_time
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
    
    def show_live_union(self, video1_path, video2_path):
        """Original method for showing just the union"""
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        # Get video properties for proper timing
        fps = cap1.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        while True:
            start_time = time.time()
            
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Create union frame
            h2, w2 = frame2.shape[:2]
            union_w, union_h = w2 + 200, h2 + 200
            union_frame = np.zeros((union_h, union_w, 3), dtype=np.uint8)
            
            # Place frame2
            union_frame[:h2, :w2] = frame2
            
            # Warp and add frame1
            warped1 = cv2.warpPerspective(frame1, self.homography, (union_w, union_h))
            union_frame = cv2.bitwise_or(union_frame, warped1)
            
            cv2.imshow('Camera Views Union', union_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Maintain proper frame rate
            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
    
    def get_union_frame(self, frame1, frame2):
        if frame1 is None or frame2 is None:
            return None
        warped_frame2 = cv2.warpPerspective(frame2, self.homography, (frame1.shape[1], frame1.shape[0]))
        union_frame = cv2.addWeighted(frame1, 0.5, warped_frame2, 0.5, 0)
        return union_frame

if __name__ == "__main__":
    union = ViewUnion()
    union.load_homography()
    
    # Show verification view
    union.show_verification_view('video1.mp4', 'video2.mp4')
    
    # Create union video
    # union.create_union_video('video1.mp4', 'video2.mp4')