from ultralytics import YOLO
import numpy as np

class YOLOv8Detector:
    def __init__(self, model_path='yolov8n.pt', device='cpu', conf=0.3):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def detect(self, image):
        """
        Run YOLOv8 detection on an image.
        Returns a list of dicts: {'bbox': (x, y, w, h), 'conf': float, 'cls': int}
        """
        results = self.model(image, device=self.device, conf=self.conf)
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