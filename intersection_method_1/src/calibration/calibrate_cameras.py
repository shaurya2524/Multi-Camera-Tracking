import cv2
import numpy as np
import glob
import os

# Settings
chessboard_size = (9, 6)  # Number of inner corners per chessboard row and column
square_size = 1.0  # Set this to the real size of a square (e.g., in cm or m)

# Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

def calibrate_camera(image_folder, output_prefix):
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(os.path.join(image_folder, '*.jpg'))
    if not images:
        print(f"No images found in {image_folder}")
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(100)
    cv2.destroyAllWindows()

    if not objpoints:
        print("No chessboard corners found!")
        return

    # Calibrate
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f"RMS re-projection error: {ret}")
    print(f"Camera matrix (K):\n{K}")
    print(f"Distortion coefficients:\n{dist.ravel()}")

    # Save parameters
    np.savez(f'{output_prefix}_calib.npz', K=K, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"Saved calibration to {output_prefix}_calib.npz")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calibrate a camera using chessboard images.')
    parser.add_argument('--images', type=str, required=True, help='Folder with chessboard images')
    parser.add_argument('--output', type=str, required=True, help='Prefix for output files (e.g., cam1, cam2)')
    args = parser.parse_args()
    calibrate_camera(args.images, args.output) 