import numpy as np
import cv2

def load_camera_params(calib_file):
    data = np.load(calib_file)
    K = data['K']
    dist = data['dist']
    rvecs = data['rvecs']
    tvecs = data['tvecs']
    return K, dist, rvecs, tvecs

def get_extrinsic_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.hstack((R, tvec.reshape(3,1)))
    return Rt

def world_to_camera(X_world, R, t):
    # X_world: (3,) or (N,3)
    return (R @ X_world.T + t).T

def camera_to_world(X_cam, R, t):
    # X_cam: (3,) or (N,3)
    return (R.T @ (X_cam.T - t)).T

def project_points(X_world, K, R, t, dist=None):
    # X_world: (N,3)
    X_world = np.asarray(X_world, dtype=np.float32)
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3,1)
    imgpts, _ = cv2.projectPoints(X_world, rvec, tvec, K, dist)
    return imgpts.squeeze()

def transform_point_between_cameras(X, R1, t1, R2, t2):
    # X: (3,) or (N,3) in camera1 coordinates
    # Returns X in camera2 coordinates
    X_world = camera_to_world(X, R1, t1)
    X_cam2 = world_to_camera(X_world, R2, t2)
    return X_cam2 