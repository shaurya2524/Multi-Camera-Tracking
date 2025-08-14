import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict
import torchreid

class PersonDetector:
    def __init__(self, model_size='n', confidence_threshold=0.8):
        """
        Initialize YOLOv8 person detector
        
        Args:
            model_size: 'n', 's', 'm', 'l', 'x' for different model sizes
            confidence_threshold: minimum confidence for detections
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DEBUG] PersonDetector using device: {self.device}")
        self.model = YOLO(f"yolov8{model_size}.pt").to(self.device)
        self.confidence_threshold = confidence_threshold
        self.target_class_id = 0  # person class in COCO
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame
        
        Args:
            frame: input image
            
        Returns:
            List of detection dictionaries with bbox, confidence, and features
        """
        results = self.model(frame)[0]
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                if int(box.cls[0]) == self.target_class_id:
                    conf = float(box.conf[0])
                    if conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Extract person crop for feature extraction
                        person_crop = frame[y1:y2, x1:x2]
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'area': (x2 - x1) * (y2 - y1),
                            'crop': person_crop
                        }
                        detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       color=(0, 255, 0)) -> np.ndarray:
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: input image
            detections: list of detection dictionaries
            color: BGR color for bounding boxes
            
        Returns:
            annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Person {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated_frame

# This will download the model weights to ~/.torchreid
torchreid.models.show_avai_models()
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1041,  # Number of classes in MSMT17, but not needed for feature extraction
    pretrained=True
)
torchreid.utils.load_pretrained_weights(model, 'osnet_x1_0_msmt17')
print("Model downloaded and loaded!")