import cv2
import numpy as np
import json

class PointSelector:
    def __init__(self):
        self.points1 = []
        self.points2 = []
        self.current_frame1 = None
        self.current_frame2 = None
        
    def mouse_callback1(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points1.append((x, y))
            cv2.circle(self.current_frame1, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.current_frame1, f"{len(self.points1)}", (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Video1 - Select Points', self.current_frame1)
    
    def mouse_callback2(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points2.append((x, y))
            cv2.circle(self.current_frame2, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(self.current_frame2, f"{len(self.points2)}", (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('Video2 - Select Points', self.current_frame2)
    
    def select_points(self, video1_path, video2_path):
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        # Get first frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error reading videos")
            return
        
        self.current_frame1 = frame1.copy()
        self.current_frame2 = frame2.copy()
        
        cv2.imshow('Video1 - Select Points', self.current_frame1)
        cv2.imshow('Video2 - Select Points', self.current_frame2)
        
        cv2.setMouseCallback('Video1 - Select Points', self.mouse_callback1)
        cv2.setMouseCallback('Video2 - Select Points', self.mouse_callback2)
        
        print("Click on matching stationary points in both videos")
        print("Press 's' to save points, 'q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if len(self.points1) >= 4 and len(self.points1) == len(self.points2):
                    # Save points to file
                    data = {
                        'points1': self.points1,
                        'points2': self.points2
                    }
                    with open('matching_points.json', 'w') as f:
                        json.dump(data, f)
                    print(f"Saved {len(self.points1)} matching points")
                    break
                else:
                    print("Need at least 4 matching points in both videos")
            elif key == ord('q'):
                break
        
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    selector = PointSelector()
    selector.select_points('video1.mp4', 'video2.mp4')