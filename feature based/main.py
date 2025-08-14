#!/usr/bin/env python3
"""
Main runner script for the Multi-Camera Tracking System

# For better tracking, try these config values:
#   "detection": {"confidence_threshold": 0.3}
#   "tracking": {"max_disappeared": 50}
#   "association": {"similarity_threshold": 0.4}
"""
import argparse

import json
import os
import sys

# Import system and components
from multi_camera_system import MultiCameraTrackingSystem
from detection import PersonDetector
from hybrid_tracker import HybridTracker
from feature_extractor import FeatureExtractorFactory
from cross_camera_association import CrossCameraAssociation

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_default_config(config_path: str):
    """Create default configuration file"""
    default_config = {
        "cameras": [
            {
                "source": "D:/ANITUM/cam1.mp4",
                "width": 640,
                "height": 480,
                "fps": 30,
                "name": "Camera 1"
            },
            {
                "source": "D:/ANITUM/cam5.mp4",
                "width": 640,
                "height": 480,
                "fps": 30,
                "name": "Camera 2"
            }
        ],
        "detection": {
            "model_size": "n",
            "confidence_threshold": 0.5
        },
        "tracking": {
            "max_disappeared": 30,
            "max_distance": 100
        },
        "features": {
            "extractor_type": "resnet50"
        },
        "association": {
            "similarity_threshold": 0.6,
            "time_window": 300,
            "max_gallery_size": 1000
        },
        "visualization": {
            "display_size": [1920, 1080],
            "show_fps": True,
            "show_global_ids": True
        }
    }
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    print(f"Default config created at {config_path}")
    return default_config

def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Tracking System Runner")
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    # Load or create config
    if not os.path.exists(args.config):
        config = create_default_config(args.config)
    else:
        config = load_config(args.config)

    # --- Instantiate components from config ---
    camera_configs = config["cameras"]
    det_cfg = config["detection"]
    track_cfg = config["tracking"]
    feat_cfg = config["features"]
    assoc_cfg = config["association"]
    vis_cfg = config.get("visualization", {})

    # Detector
    detector = PersonDetector(
        model_size=det_cfg.get("model_size", "n"),
        confidence_threshold=det_cfg.get("confidence_threshold", 0.5)
    )
    # Feature extractor
    feature_extractor = FeatureExtractorFactory.create_extractor(
        feat_cfg.get("extractor_type", "resnet50")
    )
    # Association
    cross_camera_association = CrossCameraAssociation(
        similarity_threshold=assoc_cfg.get("similarity_threshold", 0.6),
        time_window=assoc_cfg.get("time_window", 300),
        max_gallery_size=assoc_cfg.get("max_gallery_size", 1000)
    )
    # Trackers (one per camera)
    single_trackers = {}
    for i, _ in enumerate(camera_configs):
        single_trackers[i] = HybridTracker(
            max_disappeared=track_cfg.get("max_disappeared", 30),
            max_distance=track_cfg.get("max_distance", 100)
        )

    # --- Patch MultiCameraTrackingSystem to accept external components ---
    # If not possible, fallback to default constructor and warn user
    try:
        system = MultiCameraTrackingSystem(
            camera_configs,
            feature_extractor_type=feat_cfg.get("extractor_type", "resnet50")
        )
        # Patch system components
        system.detector = detector
        system.feature_extractor = feature_extractor
        system.cross_camera_association = cross_camera_association
        system.single_trackers = single_trackers
    except Exception as e:
        print("[Warning] Could not inject all config parameters into MultiCameraTrackingSystem:", e)
        print("Falling back to default parameters.")
        system = MultiCameraTrackingSystem(camera_configs)

    # --- Run the pipeline ---
    try:
        system.initialize_cameras()
        system.start_tracking()
        display_size = tuple(vis_cfg.get("display_size", [1920, 1080]))
        system.visualize_results(display_size=display_size)
    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        system.stop_tracking()
        print("System stopped")

if __name__ == "__main__":
    main() 