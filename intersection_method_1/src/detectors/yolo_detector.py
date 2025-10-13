#!/usr/bin/env python3
"""
YOLO Object Detector Module

This module provides a wrapper for YOLOv8/YOLO11 object detection models
using the Ultralytics library. It handles model loading, GPU acceleration,
and detection processing.

Author: Multi-Camera Tracking System
"""

import torch
import numpy as np
from ultralytics import YOLO


class YOLOv8Detector:
    def __init__(self, model_path='yolo11n.pt', confidence_threshold=0.6):
        """
        Initialize YOLO detector
        
        Args:
            model_path (str): Path to YOLO model file
            confidence_threshold (float): Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Initializing YOLO detector on device: {self.device}")
        
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            # Fallback to default model
            try:
                self.model = YOLO('yolov8n.pt')
                self.model.to(self.device)
                print("Loaded fallback model: yolov8n.pt")
            except Exception as e2:
                raise RuntimeError(f"Failed to load any YOLO model: {e2}")

    def detect(self, frame):
        """
        Detect objects in frame using YOLO model
        
        Args:
            frame (np.ndarray): Input image frame
            
        Returns:
            list: List of detection dictionaries with keys:
                - bbox: (x, y, width, height) bounding box
                - conf: confidence score
                - cls: class ID
        """
        try:
            results = self.model.predict(
                frame, 
                device=self.device,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Convert to (x, y, width, height) format
                        bbox = (
                            int(x1), 
                            int(y1), 
                            int(x2 - x1), 
                            int(y2 - y1)
                        )
                        
                        detections.append({
                            'bbox': bbox,
                            'conf': confidence,
                            'cls': class_id
                        })
            
            return detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return []
    
    def set_confidence_threshold(self, threshold):
        """
        Update confidence threshold
        
        Args:
            threshold (float): New confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"Updated confidence threshold to: {self.confidence_threshold}")
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information including device and class names
        """
        return {
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'model_type': type(self.model).__name__,
            'class_names': getattr(self.model, 'names', {})
        } 
