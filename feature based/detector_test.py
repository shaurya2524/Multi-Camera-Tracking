import cv2
from ultralytics import YOLO
import torch

# ✅ Force GPU usage (if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolov8n.pt").to(device)  # Use 's', 'm', 'l', 'x' for stronger models

TARGET_CLASS_ID = 0  # person class

# Load two video files
cap1 = cv2.VideoCapture(r"D:\ANITUM\cam1.mp4")
cap2 = cv2.VideoCapture(r"D:\ANITUM\cam5.mp4")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Resize frames to same height
    height = 480
    frame1 = cv2.resize(frame1, (int(frame1.shape[1] * height / frame1.shape[0]), height))
    frame2 = cv2.resize(frame2, (int(frame2.shape[1] * height / frame2.shape[0]), height))

    # Run detection on both frames
    results1 = model(frame1)[0]
    results2 = model(frame2)[0]

    # Draw detections for frame 1
    for box in results1.boxes:
        if int(box.cls[0]) == TARGET_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Person {conf:.2f}"
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame1, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw detections for frame 2
    for box in results2.boxes:
        if int(box.cls[0]) == TARGET_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Person {conf:.2f}"
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame2, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Combine both frames side by side
    combined = cv2.hconcat([frame1, frame2])

    # Show combined frame
    cv2.imshow("YOLOv8 Human Detection - Side by Side", combined)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
