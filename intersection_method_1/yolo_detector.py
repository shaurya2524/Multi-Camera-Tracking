from ultralytics import YOLO
import numpy as np
import torch
class YOLOv8Detector:
    def __init__(self):
        # Load model and send to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        self.model = YOLO('yolov8n.pt')
        self.model.to(device)
        self.device = device

    def detect(self, frame):
        results = self.model.predict(frame, device=self.device,conf=0.6)  # Force device each call
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    'conf': conf,
                    'cls': cls
                })
        return detections 