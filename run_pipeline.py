
"""
Complete pipeline for camera view union
Usage: python run_pipeline.py video1.mp4 video2.mp4
"""

import sys
import os
import cv2
import numpy as np
from point_selector import PointSelector
from homography_calculator import HomographyCalculator
from view_union import ViewUnion
from camera_utils import load_camera_params, get_extrinsic_matrix, project_points, transform_point_between_cameras

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_pipeline.py video1.mp4 video2.mp4")
        sys.exit(1)
    
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    
    # Check if videos exist
    if not os.path.exists(video1_path) or not os.path.exists(video2_path):
        print("Error: Video files not found")
        sys.exit(1)
    
    print("=== Camera View Union Pipeline ===")
    
    # Step 0: Load camera calibration (if available)
    cam1_calib = 'cam1_calib.npz'
    cam2_calib = 'cam2_calib.npz'
    if os.path.exists(cam1_calib) and os.path.exists(cam2_calib):
        print("\nStep 0: Load camera calibration parameters")
        K1, dist1, rvecs1, tvecs1 = load_camera_params(cam1_calib)
        K2, dist2, rvecs2, tvecs2 = load_camera_params(cam2_calib)
        # Use first view for extrinsics (single calibration)
        R1, t1 = cv2.Rodrigues(rvecs1[0])[0], tvecs1[0].reshape(3,1)
        R2, t2 = cv2.Rodrigues(rvecs2[0])[0], tvecs2[0].reshape(3,1)
        print(f"Loaded cam1: K=\n{K1}\nR=\n{R1}\nt=\n{t1.flatten()}")
        print(f"Loaded cam2: K=\n{K2}\nR=\n{R2}\nt=\n{t2.flatten()}")
        # Example: transform a point from cam1 to cam2
        example_pt = np.array([[0,0,0]], dtype=np.float32)  # origin in world
        imgpt1 = project_points(example_pt, K1, R1, t1, dist1)
        imgpt2 = project_points(example_pt, K2, R2, t2, dist2)
        print(f"World origin projects to cam1: {imgpt1}, cam2: {imgpt2}")
    else:
        print("\n[Warning] Camera calibration files not found. Run calibrate_cameras.py for each camera.")
    
    # Step 1: Select matching points
    print("\nStep 1: Select matching points")
    if not os.path.exists('matching_points.json'):
        print("Select at least 4 matching stationary points in both videos")
        selector = PointSelector()
        selector.select_points(video1_path, video2_path)
    else:
        print("Using existing matching points")
    
    # Step 2: Calculate homography
    print("\nStep 2: Calculate homography matrix")
    calc = HomographyCalculator()
    homography = calc.calculate_homography()
    
    if homography is None:
        print("Failed to calculate homography. Please select better points.")
        sys.exit(1)
    
    # Step 3: Create union
    print("\nStep 3: Create camera views union")
    union = ViewUnion()
    union.load_homography()
    
    # Show verification view with original videos side by side
    print("Showing verification view with original videos and union")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 'SPACE' to pause/resume")
    print("- Press 'r' to restart videos")
    union.show_verification_view(video1_path, video2_path)
    
    # Ask if user wants to save video
    save_video = input("\nSave union video? (y/n): ").lower().strip()
    if save_video == 'y':
        output_name = input("Output filename (default: union_output.mp4): ").strip()
        if not output_name:
            output_name = 'union_output.mp4'
        
        print(f"Creating union video: {output_name}")
        union.create_union_video(video1_path, video2_path, output_name)
    
    print("\nPipeline completed!")

if __name__ == "__main__":
    main()