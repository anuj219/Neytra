# ai/detector.py

import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition

CONF_THRESHOLD = 0.5
MODEL_PATH ="D:\\Neytra\\models\\bestv4.pt"  # ← FIXED: Removed extra quotes

# Global model instance (loaded once, reused)
yolo_model = None

def load_yolo_model():
    """Load YOLO model once at startup"""
    global yolo_model
    if yolo_model is not None:
        return yolo_model
    
    try:
        print(f"[YOLO] Loading model from: {MODEL_PATH}")
        yolo_model = YOLO(MODEL_PATH)
        print("[YOLO] ✅ Model loaded successfully")
        return yolo_model
    except Exception as e:
        print(f"[YOLO] ❌ Failed to load model: {e}")
        yolo_model = None
        return None


def detect_yolo(frame, fast_mode=False):
    """
    YOLO object detection
    
    Args:
        frame: Input image
        fast_mode: If True, use lower resolution for faster detection (quickscan)
    """
    detections = []
    
    if yolo_model is None:
        print("[YOLO] Model not loaded, attempting to load...")
        load_yolo_model()
        if yolo_model is None:
            return detections

    try:
        # Adjust image size based on mode
        imgsz = 320 if fast_mode else 640
        
        results = yolo_model(frame, imgsz=imgsz, verbose=False)
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
        
        print(f"[YOLO] Detected {len(detections)} objects")
        
    except Exception as e:
        print(f"[YOLO ERROR] {e}")
    
    return detections


def detect_faces_fallback(frame):
    """Fallback face detection using face_recognition library"""
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
