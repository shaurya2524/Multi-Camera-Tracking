import cv2
import numpy as np
import json

class HomographyCalculator:
    def __init__(self):
        self.homography = None
        
    def load_points(self, points_file='matching_points.json'):
        with open(points_file, 'r') as f:
            data = json.load(f)
        return np.array(data['points1']), np.array(data['points2'])
    
    def calculate_homography(self, points_file='matching_points.json'):
        pts1, pts2 = self.load_points(points_file)
        
        # Calculate homography matrix (transforms pts1 to pts2 coordinate system)
        self.homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        if self.homography is not None:
            print("Homography matrix calculated successfully")
            print(f"Inliers: {np.sum(mask)}/{len(pts1)}")
            print("Transformation matrix (H):")
            print(self.homography)
            
            # Save homography
            np.save('homography_matrix.npy', self.homography)
            print("Homography saved to homography_matrix.npy")
        else:
            print("Failed to calculate homography")
            
        return self.homography
    
    def test_homography(self, video1_path, video2_path):
        if self.homography is None:
            self.homography = np.load('homography_matrix.npy')
        
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            # Warp frame1 to frame2's perspective
            h, w = frame2.shape[:2]
            warped = cv2.warpPerspective(frame1, self.homography, (w, h))
            
            # Show results
            cv2.imshow('Original Video1', frame1)
            cv2.imshow('Original Video2', frame2)
            cv2.imshow('Warped Video1', warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        cap1.release()
        cap2.release()

if __name__ == "__main__":
    calc = HomographyCalculator()
    H = calc.calculate_homography()
    if H is not None:
        calc.test_homography('video1.mp4', 'video2.mp4')