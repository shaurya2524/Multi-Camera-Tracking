import numpy as np
import cv2

class YoloMultiCameraTracker:
    def __init__(self, homography_path='homography_matrix.npy', max_disappeared=10, max_distance=100):
        self.homography = np.load(homography_path)
        self.tracks = {}  # id: {'last_position': (x, y), 'disappeared': int, ...}
        self.next_id = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def transform_point(self, point):
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography)
        return tuple(transformed[0][0].astype(int))

    def update(self, detections_cam1, detections_cam2):
        # Map cam1 detections to union region
        union_detections = []
        for det in detections_cam1:
            center = (det['bbox'][0] + det['bbox'][2] // 2, det['bbox'][1] + det['bbox'][3] // 2)
            union_center = self.transform_point(center)
            union_detections.append({'center': union_center, 'bbox': det['bbox'], 'camera': 1})
        for det in detections_cam2:
            center = (det['bbox'][0] + det['bbox'][2] // 2, det['bbox'][1] + det['bbox'][3] // 2)
            union_detections.append({'center': center, 'bbox': det['bbox'], 'camera': 2})

        # Match detections to existing tracks
        matched_ids = set()
        for det in union_detections:
            best_id = None
            best_dist = float('inf')
            for track_id, track in self.tracks.items():
                if track_id in matched_ids:
                    continue
                dist = np.linalg.norm(np.array(det['center']) - np.array(track['last_position']))
                if dist < self.max_distance and dist < best_dist:
                    best_dist = dist
                    best_id = track_id
            if best_id is not None:
                self.tracks[best_id]['last_position'] = det['center']
                self.tracks[best_id]['disappeared'] = 0
                self.tracks[best_id]['camera'] = det['camera']
                self.tracks[best_id]['bbox'] = det['bbox']
                matched_ids.add(best_id)
            else:
                self.tracks[self.next_id] = {
                    'last_position': det['center'],
                    'disappeared': 0,
                    'camera': det['camera'],
                    'bbox': det['bbox']
                }
                self.next_id += 1

        # Increment disappeared for unmatched tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_ids:
                track['disappeared'] += 1
                if track['disappeared'] > self.max_disappeared:
                    to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]

    def draw_tracks(self, frame, camera_id, union_frame=None):
        for track_id, track in self.tracks.items():
            if track['camera'] == camera_id and frame is not None:
                x, y, w, h = track['bbox']
                center = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if union_frame is not None:
                center = track['last_position']
                cv2.circle(union_frame, center, 8, (0, 0, 255), -1)
                cv2.putText(union_frame, f'ID: {track_id}', (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cam_color = (255, 0, 0) if track['camera'] == 1 else (0, 255, 255)
                cv2.putText(union_frame, f'Cam{track["camera"]}', (center[0] + 10, center[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cam_color, 1) 