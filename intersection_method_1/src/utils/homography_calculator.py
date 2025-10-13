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
    
    def calculate_homography(self, points_file='matching_points.json', 
                              use_sift=False, video1_path=None, video2_path=None):
        """
        Calculate homography using either:
        - matching_points.json (default), or
        - SIFT feature matching on the first frames of two videos (if use_sift=True)
        """
        if use_sift:
            if video1_path is None or video2_path is None:
                raise ValueError("For SIFT mode, you must provide video1_path and video2_path")
            
            # Grab first frames from both videos
            cap1 = cv2.VideoCapture(video1_path)
            cap2 = cv2.VideoCapture(video2_path)
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            cap1.release()
            cap2.release()
            
            if not (ret1 and ret2):
                raise ValueError("Could not read frames from one or both videos")
            
            # Convert to grayscale
            img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # SIFT feature detection
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # FLANN matcher
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # Lowe’s ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) < 4:
                print("Not enough good matches found for SIFT")
                return None

            # --- Show matches before homography ---
            match_vis = cv2.drawMatches(
                frame1, kp1, frame2, kp2, good, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imshow("SIFT Matches", match_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Extract matched points
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        else:
            # Load manually provided points
            pts1, pts2 = self.load_points(points_file)

        # Calculate homography
        self.homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        if self.homography is not None:
            print("Homography matrix calculated successfully")
            print(f"Inliers: {np.sum(mask)}/{len(mask)}")
            print("Transformation matrix (H):")
            print(self.homography)

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
            h, w = frame2.shape[:2]
            warped = cv2.warpPerspective(frame1, self.homography, (w, h))
            
            cv2.imshow('Original Video1', frame1)
            cv2.imshow('Original Video2', frame2)
            cv2.imshow('Warped Video1', warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        cap1.release()
        cap2.release()

if __name__ == "__main__":
    calc = HomographyCalculator()
    
    # Option 1: JSON points
    # H = calc.calculate_homography()

    # Option 2: SIFT on first frames of the two videos (with matches visualization)
    H = calc.calculate_homography(use_sift=True, video1_path="video1.mp4", video2_path="video2.mp4")

    if H is not None:
        calc.test_homography('video1.mp4', 'video2.mp4')
