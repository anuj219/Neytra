# ai/detector.py

import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition

CONF_THRESHOLD = 0.5
MODEL_PATH = 'C:\\Users\\anujv\\OneDrive\\Desktop\\Programming\\Codes\\python\\Face Recognition\\Neytra\\Object\\Neytra-Obj_Detection\\models\\bestLatest.pt'  # load your best.pt here if available

yolo_model = None
try:
    if MODEL_PATH:
        yolo_model = YOLO(MODEL_PATH)
        print("YOLO model loaded successfully.")
except:
    yolo_model = None


def detect_yolo(frame):
    detections = []
    if yolo_model is None:
        return detections

    results = yolo_model(frame, imgsz=320, verbose=False)
    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None:
        return detections

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = r.names[cls]

        if conf < CONF_THRESHOLD:
            continue

        detections.append({
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "confidence": conf
        })

    return detections


def detect_faces_fallback(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locs = face_recognition.face_locations(rgb)
    dets = []
    for (top, right, bottom, left) in face_locs:
        dets.append({
            "label": "person",
            "bbox": [left, top, right, bottom],
            "confidence": 0.9
        })
    return dets
