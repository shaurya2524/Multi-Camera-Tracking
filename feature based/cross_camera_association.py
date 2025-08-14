import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Set
import time
import cv2

class CrossCameraAssociation:
    """Cross-camera association for multi-camera tracking"""
    
    def __init__(self, similarity_threshold=0.6, time_window=300, max_gallery_size=1000):
        """
        Initialize cross-camera association
        
        Args:
            similarity_threshold: Minimum similarity for association
            time_window: Time window for association (seconds)
            max_gallery_size: Maximum number of tracks in gallery per camera
        """
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        self.max_gallery_size = max_gallery_size
        
        # Global track management
        self.global_tracks = {}  # global_id -> GlobalTrack
        self.camera_tracks = defaultdict(dict)  # camera_id -> {local_id -> global_id}
        self.next_global_id = 0
        
        # Track galleries for each camera
        self.track_galleries = defaultdict(lambda: deque(maxlen=max_gallery_size))
        
        # Similarity cache
        self.similarity_cache = {}
        
        self.reid_lost_tracks = {}  # global_id -> (features, last_seen, bboxes)
        self.reid_lost_window = 200  # frames
        
    def update_camera_tracks(self, camera_id: int, tracklets: List[Dict]):
        """
        Update tracks for a specific camera
        
        Args:
            camera_id: Camera identifier
            tracklets: List of tracklets from single-camera tracking
        """
        current_time = time.time()
        
        # Process each tracklet
        for tracklet in tracklets:
            local_id = tracklet['track_id']
            
            # Check if this local track is already associated
            if local_id in self.camera_tracks[camera_id]:
                global_id = self.camera_tracks[camera_id][local_id]
                self._update_global_track(global_id, camera_id, tracklet, current_time)
            else:
                # Try to associate with existing global tracks
                global_id = self._associate_tracklet(camera_id, tracklet, current_time)
                if global_id is None:
                    # Try advanced re-identification with recently lost global tracks
                    global_id = self._reid_lost_tracklet(tracklet, current_time)
                if global_id is None:
                    # Create new global track
                    global_id = self._create_global_track(camera_id, tracklet, current_time)
                
                self.camera_tracks[camera_id][local_id] = global_id
        
        # Update track gallery
        self._update_track_gallery(camera_id, tracklets)
        
        # Clean up old tracks
        self._cleanup_old_tracks(current_time)
        self._cleanup_lost_tracks(current_time)
    
    def _associate_tracklet(self, camera_id: int, tracklet: Dict, current_time: float) -> Optional[int]:
        """
        Associate a tracklet with existing global tracks
        
        Args:
            camera_id: Camera identifier
            tracklet: Tracklet to associate
            current_time: Current timestamp
            
        Returns:
            Global track ID if association found, None otherwise
        """
        if tracklet['features'] is None:
            return None
        
        best_global_id = None
        best_similarity = 0.0
        
        # Compare with all active global tracks from other cameras
        for global_id, global_track in self.global_tracks.items():
            # Skip if this global track is already active in this camera
            if camera_id in global_track.active_cameras:
                continue
            
            # Check temporal constraint
            if current_time - global_track.last_seen > self.time_window:
                continue
            
            # Compute similarity
            similarity = self._compute_tracklet_similarity(tracklet, global_track)
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_global_id = global_id
        
        return best_global_id
    
    def _compute_tracklet_similarity(self, tracklet: Dict, global_track: 'GlobalTrack') -> float:
        """
        Compute similarity between tracklet and global track
        
        Args:
            tracklet: Tracklet features
            global_track: Global track object
            
        Returns:
            Similarity score
        """
        if tracklet['features'] is None:
            return 0.0
        
        # Get representative features from global track (last 10)
        global_features = global_track.get_representative_features(num_last=10)
        if global_features is None:
            return 0.0
        
        # Use average of last 3 features from tracklet if available
        if 'features_history' in tracklet and len(tracklet['features_history']) >= 1:
            tracklet_features = np.mean(tracklet['features_history'][-3:], axis=0)
        else:
            tracklet_features = tracklet['features']
        
        # Normalize features
        tracklet_norm = tracklet_features / (np.linalg.norm(tracklet_features) + 1e-8)
        global_norm = global_features / (np.linalg.norm(global_features) + 1e-8)
        
        similarity = np.dot(tracklet_norm, global_norm)
        
        # Apply temporal decay
        time_diff = time.time() - global_track.last_seen
        temporal_weight = max(0, 1 - time_diff / self.time_window)
        
        return similarity * temporal_weight
    
    def _create_global_track(self, camera_id: int, tracklet: Dict, current_time: float) -> int:
        """
        Create a new global track
        
        Args:
            camera_id: Camera identifier
            tracklet: Initial tracklet
            current_time: Current timestamp
            
        Returns:
            Global track ID
        """
        global_id = self.next_global_id
        self.next_global_id += 1
        
        global_track = GlobalTrack(global_id, camera_id, tracklet, current_time)
        self.global_tracks[global_id] = global_track
        
        return global_id
    
    def _update_global_track(self, global_id: int, camera_id: int, tracklet: Dict, current_time: float):
        """
        Update existing global track
        
        Args:
            global_id: Global track identifier
            camera_id: Camera identifier
            tracklet: Updated tracklet
            current_time: Current timestamp
        """
        if global_id in self.global_tracks:
            self.global_tracks[global_id].update(camera_id, tracklet, current_time)
    
    def _update_track_gallery(self, camera_id: int, tracklets: List[Dict]):
        """
        Update track gallery for cross-camera matching
        
        Args:
            camera_id: Camera identifier
            tracklets: List of tracklets
        """
        # Add tracklets to gallery
        for tracklet in tracklets:
            if tracklet['features'] is not None:
                gallery_entry = {
                    'camera_id': camera_id,
                    'track_id': tracklet['track_id'],
                    'features': tracklet['features'],
                    'timestamp': time.time(),
                    'bbox': tracklet['bbox']
                }
                self.track_galleries[camera_id].append(gallery_entry)
    
    def _cleanup_old_tracks(self, current_time: float):
        """
        Clean up old and inactive tracks
        
        Args:
            current_time: Current timestamp
        """
        # Remove old global tracks
        global_ids_to_remove = []
        for global_id, global_track in self.global_tracks.items():
            if current_time - global_track.last_seen > self.time_window * 2:
                global_ids_to_remove.append(global_id)
        
        for global_id in global_ids_to_remove:
            del self.global_tracks[global_id]
            
            # Remove from camera tracks
            for camera_id in self.camera_tracks:
                local_ids_to_remove = []
                for local_id, gid in self.camera_tracks[camera_id].items():
                    if gid == global_id:
                        local_ids_to_remove.append(local_id)
                
                for local_id in local_ids_to_remove:
                    del self.camera_tracks[camera_id][local_id]
            
            # Move lost global tracks to reid_lost_tracks
            if global_id in self.reid_lost_tracks:
                global_track = self.global_tracks[global_id]
                features = global_track.get_representative_features(num_last=10)
                bboxes = global_track.get_current_bboxes()
                self.reid_lost_tracks.pop(global_id, None)
    
    def get_global_tracks(self) -> Dict[int, Dict]:
        """
        Get all active global tracks
        
        Returns:
            Dictionary of global tracks
        """
        current_time = time.time()
        active_tracks = {}
        
        for global_id, global_track in self.global_tracks.items():
            if current_time - global_track.last_seen <= self.time_window:
                active_tracks[global_id] = {
                    'global_id': global_id,
                    'cameras': list(global_track.active_cameras.keys()),
                    'last_seen': global_track.last_seen,
                    'track_length': global_track.track_length,
                    'features': global_track.get_representative_features(),
                    'bboxes': global_track.get_current_bboxes()
                }
        
        return active_tracks
    
    def query_similar_tracks(self, query_features: np.ndarray, camera_id: int, 
                           top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Query similar tracks across all cameras
        
        Args:
            query_features: Query feature vector
            camera_id: Query camera ID
            top_k: Number of top results to return
            
        Returns:
            List of (global_id, similarity) tuples
        """
        similarities = []
        
        for global_id, global_track in self.global_tracks.items():
            # Skip tracks from same camera
            if camera_id in global_track.active_cameras:
                continue
            
            global_features = global_track.get_representative_features()
            if global_features is not None:
                similarity = self._compute_feature_similarity(query_features, global_features)
                similarities.append((global_id, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _compute_feature_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            Cosine similarity
        """
        if feat1 is None or feat2 is None:
            return 0.0
        
        # Normalize features
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        return np.dot(feat1_norm, feat2_norm)

    def _reid_lost_tracklet(self, tracklet: Dict, current_time: float) -> int:
        """Try to re-identify a tracklet with recently lost global tracks"""
        if tracklet['features'] is None:
            return None
        best_global_id = None
        best_similarity = 0.0
        for global_id, (features, last_seen, bboxes) in self.reid_lost_tracks.items():
            if current_time - last_seen > self.reid_lost_window:
                continue
            # Compute similarity
            feat_norm = features / (np.linalg.norm(features) + 1e-8)
            tracklet_norm = tracklet['features'] / (np.linalg.norm(tracklet['features']) + 1e-8)
            similarity = np.dot(feat_norm, tracklet_norm)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_global_id = global_id
        if best_global_id is not None:
            # Reuse the old global ID
            print(f"[REID] Re-identified tracklet as global ID {best_global_id} (sim={best_similarity:.2f})")
            # Remove from lost buffer (will be reactivated)
            self.reid_lost_tracks.pop(best_global_id, None)
        return best_global_id

    def _cleanup_lost_tracks(self, current_time: float):
        # Remove lost tracks that are too old
        to_remove = []
        for global_id, (features, last_seen, bboxes) in self.reid_lost_tracks.items():
            if current_time - last_seen > self.reid_lost_window:
                to_remove.append(global_id)
        for gid in to_remove:
            self.reid_lost_tracks.pop(gid, None)

class GlobalTrack:
    """Global track across multiple cameras"""
    
    def __init__(self, global_id: int, camera_id: int, initial_tracklet: Dict, timestamp: float):
        """
        Initialize global track
        
        Args:
            global_id: Global track identifier
            camera_id: Initial camera ID
            initial_tracklet: Initial tracklet
            timestamp: Initial timestamp
        """
        self.global_id = global_id
        self.created_time = timestamp
        self.last_seen = timestamp
        self.track_length = 0
        
        # Camera-specific information
        self.active_cameras = {camera_id: timestamp}
        self.camera_tracklets = defaultdict(list)
        self.camera_features = defaultdict(list)
        self.features_history = []
        
        # Add initial tracklet
        self.update(camera_id, initial_tracklet, timestamp)
    
    def update(self, camera_id: int, tracklet: Dict, timestamp: float):
        """
        Update global track with new tracklet
        
        Args:
            camera_id: Camera identifier
            tracklet: Updated tracklet
            timestamp: Current timestamp
        """
        self.last_seen = timestamp
        self.active_cameras[camera_id] = timestamp
        self.track_length += 1
        
        # Store tracklet and features
        self.camera_tracklets[camera_id].append(tracklet)
        if tracklet['features'] is not None:
            self.camera_features[camera_id].append(tracklet['features'])
            self.features_history.append(tracklet['features'])
        
        # Keep only recent tracklets (memory management)
        max_tracklets = 50
        if len(self.camera_tracklets[camera_id]) > max_tracklets:
            self.camera_tracklets[camera_id] = self.camera_tracklets[camera_id][-max_tracklets:]
            self.camera_features[camera_id] = self.camera_features[camera_id][-max_tracklets:]
    
    def get_representative_features(self, num_last=20) -> Optional[np.ndarray]:
        """
        Get representative features for this global track
        
        Returns:
            Representative feature vector
        """
        all_features = []
        
        # Collect features from all cameras
        for camera_id, features_list in self.camera_features.items():
            all_features.extend(features_list)
        
        if not all_features:
            return None
        
        # Use average of recent features
        recent_features = all_features[-num_last:]  # Last num_last features
        return np.mean(recent_features, axis=0)
    
    def get_current_bboxes(self) -> Dict[int, List[int]]:
        """
        Get current bounding boxes for each camera
        
        Returns:
            Dictionary of camera_id -> bbox
        """
        current_bboxes = {}
        
        for camera_id, tracklets in self.camera_tracklets.items():
            if tracklets:
                current_bboxes[camera_id] = tracklets[-1]['bbox']
        
        return current_bboxes
    
    def is_active_in_camera(self, camera_id: int, current_time: float, timeout: float = 10) -> bool:
        """
        Check if track is active in specific camera
        
        Args:
            camera_id: Camera identifier
            current_time: Current timestamp
            timeout: Timeout for considering track inactive
            
        Returns:
            True if track is active in camera
        """
        if camera_id not in self.active_cameras:
            return False
        
        return current_time - self.active_cameras[camera_id] <= timeout

class ReRankingOptimizer:
    """Re-ranking optimizer for improving cross-camera associations"""
    
    def __init__(self, k1=20, k2=6, lambda_value=0.3):
        """
        Initialize re-ranking optimizer
        
        Args:
            k1: Number of nearest neighbors for first ranking
            k2: Number of nearest neighbors for second ranking
            lambda_value: Weight for original distance
        """
        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value
    
    def re_rank(self, query_features: np.ndarray, gallery_features: List[np.ndarray]) -> List[int]:
        """
        Re-rank gallery features based on query
        
        Args:
            query_features: Query feature vector
            gallery_features: List of gallery feature vectors
            
        Returns:
            Re-ranked indices
        """
        if not gallery_features:
            return []
        
        # Compute initial distances
        distances = []
        for gallery_feat in gallery_features:
            if gallery_feat is not None:
                dist = self._compute_distance(query_features, gallery_feat)
                distances.append(dist)
            else:
                distances.append(float('inf'))
        
        distances = np.array(distances)
        
        # Get initial ranking
        initial_ranking = np.argsort(distances)
        
        # Apply k-reciprocal re-ranking
        re_ranked_indices = self._k_reciprocal_rerank(query_features, gallery_features, distances)
        
        return re_ranked_indices
    
    def _compute_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute Euclidean distance between features
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            Euclidean distance
        """
        if feat1 is None or feat2 is None:
            return float('inf')
        
        return np.linalg.norm(feat1 - feat2)
    
    def _k_reciprocal_rerank(self, query_features: np.ndarray, 
                           gallery_features: List[np.ndarray], 
                           distances: np.ndarray) -> List[int]:
        """
        Apply k-reciprocal re-ranking
        
        Args:
            query_features: Query feature vector
            gallery_features: Gallery feature vectors
            distances: Initial distances
            
        Returns:
            Re-ranked indices
        """
        # For simplicity, return initial ranking
        # In practice, implement full k-reciprocal re-ranking algorithm
        return np.argsort(distances).tolist()