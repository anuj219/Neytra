# ai/pipeline.py

import cv2
import face_recognition
from ai.detector import detect_yolo, detect_faces_fallback, load_yolo_model
from ai.recognizer import load_database, compare_face_to_db
from ai.encounter import update_presence, mark_absent

FACE_MATCH_THRESHOLD = 0.6
PERSON_LABELS = {"person", "face", "human"}

# Load database once at startup
db = load_database()
print(f"[FACE DB] Loaded {len(db)} people: {list(db.keys())}")

# Load YOLO model once at startup
load_yolo_model()


def process_frame_scan(frame):
    """
    Standard scan mode: Normal object detection
    Used when frontend sends frame every 10 seconds
    """
    print("[PIPELINE] Running SCAN mode")
    
    # Detect objects with YOLO
    detections = detect_yolo(frame, fast_mode=False)
    
    # Fallback to face detection if no objects found
    if not detections:
        detections = detect_faces_fallback(frame)
    
    results = []
    for det in detections:
        results.append({
            "type": "object",
            "label": det["label"],
            "bbox": det["bbox"],
            "confidence": det.get("confidence", 0.0)
        })
    
    print(f"[SCAN] Found {len(results)} objects")
    return results


def process_frame_quickscan(frame):
    """
    Quick scan mode: Fast object detection for urgent scenarios
    Lower resolution, focus on obstacles and people
    """
    print("[PIPELINE] Running QUICKSCAN mode")
    
    # Use fast mode (lower resolution)
    detections = detect_yolo(frame, fast_mode=True)
    
    # Prioritize people and potential obstacles
    priority_labels = {"person", "car", "bicycle", "motorcycle", "truck", "bus"}
    
    results = []
    for det in detections:
        is_priority = det["label"].lower() in priority_labels
        results.append({
            "type": "object",
            "label": det["label"],
            "bbox": det["bbox"],
            "confidence": det.get("confidence", 0.0),
            "priority": is_priority
        })
    
    # Sort by priority
    results.sort(key=lambda x: x.get("priority", False), reverse=True)
    
    print(f"[QUICKSCAN] Found {len(results)} objects")
    return results


def process_frame_face(frame):
    """
    Face recognition mode: Identify known faces or enroll new ones
    This is the full pipeline including face recognition
    """
    print("[PIPELINE] Running FACE RECOGNITION mode")
    
    # Step 1: detect objects/persons
    detections = detect_yolo(frame, fast_mode=False)
    
    if not detections:
        detections = detect_faces_fallback(frame)
    else:
        has_person = any(det["label"].lower() in PERSON_LABELS for det in detections)
        if not has_person:
            detections.extend(detect_faces_fallback(frame))

    names_this_frame = set()
    results = []

    for det in detections:
        label = det["label"].lower()
        x1, y1, x2, y2 = det["bbox"]
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        
        if x2 <= x1 or y2 <= y1:
            continue

        if label in PERSON_LABELS:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_crop)
            face_encs = face_recognition.face_encodings(rgb_crop, face_locs)

            if not face_encs:
                results.append({
                    "type": "face",
                    "name": "unknown",
                    "bbox": det["bbox"],
                    "announce": False
                })
                continue

            face_enc = face_encs[0]
            best_name, best_dist = compare_face_to_db(face_enc, db)

            if best_name and best_dist < FACE_MATCH_THRESHOLD:
                announce = update_presence(best_name)
                names_this_frame.add(best_name)

                results.append({
                    "type": "face",
                    "name": best_name,
                    "distance": best_dist,
                    "bbox": det["bbox"],
                    "announce": announce
                })
                print(f"[FACE] Recognized: {best_name} (distance: {best_dist:.3f})")
            else:
                results.append({
                    "type": "face",
                    "name": "unknown",
                    "bbox": det["bbox"],
                    "announce": False
                })
                print(f"[FACE] Unknown person (best match: {best_name}, dist: {best_dist:.3f})")
        else:
            results.append({
                "type": "object",
                "label": det["label"],
                "bbox": det["bbox"]
            })

    mark_absent(names_this_frame)
    print(f"[FACE RECOGNITION] Found {len(results)} detections")
    return results


# Legacy function for backward compatibility
def process_frame(frame):
    """Backward compatible: defaults to face recognition mode"""
    return process_frame_face(frame)
