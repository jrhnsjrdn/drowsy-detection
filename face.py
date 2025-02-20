import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n_ori.pt")

# Define class labels (adjust based on your training dataset)
CLASS_NAMES = ["Awake", "Drowsy"]  # Adjust according to your model

# Open laptop camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame)

    # Process each detected object
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            cls_index = int(cls)
            
            # Ignore classes outside the range of CLASS_NAMES
            if cls_index < 0 or cls_index >= len(CLASS_NAMES):
                continue  # Skip this detection
            
            label = CLASS_NAMES[cls_index]  # Get class label
            confidence = float(conf)  # Confidence score
            
            # Draw bounding box
            color = (0, 255, 0) if label == "Awake" else (0, 0, 255)
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display label & confidence
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show result
    cv2.imshow("YOLOv8 Drowsiness Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
