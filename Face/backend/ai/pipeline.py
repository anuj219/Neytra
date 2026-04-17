# ai/pipeline.py

import cv2
import face_recognition
import base64
import numpy as np
import time
from ai.detector import detect_yolo, detect_faces_fallback, load_yolo_model
from ai.recognizer import load_database, compare_face_to_db
from ai.encounter import update_presence, mark_absent

FACE_MATCH_THRESHOLD = 0.48  # Stricter threshold; 0.6 was causing false positives with small DB
PERSON_LABELS = {"person", "face", "human"}
UNKNOWN_ENROLLMENT_DELAY = 10.0  # seconds a face must remain in frame before prompting enrollment
UNKNOWN_TRACKER_TTL = 5.0
unknown_face_trackers = {}

# Load database once at startup
db = load_database()
print(f"[FACE DB] Loaded {len(db)} people: {list(db.keys())}")


def reload_database():
    """Reload face database from disk (call after enrollment saves a new face)"""
    global db
    db = load_database()
    print(f"[FACE DB] Reloaded {len(db)} people: {list(db.keys())}")
    return db

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
    
    # If YOLO finds no objects or finds objects but no person-like labels,
    # fallback to direct face detection so people are not missed.
    has_person = any(det["label"].lower() in PERSON_LABELS for det in detections)
    if not detections or not has_person:
        detections.extend(detect_faces_fallback(frame))
    
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
    
    # If YOLO finds no people, fallback to face detection so important humans are not missed.
    has_person = any(det["label"].lower() in PERSON_LABELS for det in detections)
    if not detections or not has_person:
        detections.extend(detect_faces_fallback(frame))
    
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


def cleanup_unknown_trackers(ttl=UNKNOWN_TRACKER_TTL):
    now = time.time()
    stale_keys = [k for k, v in unknown_face_trackers.items() if now - v["last_seen"] > ttl]
    for key in stale_keys:
        del unknown_face_trackers[key]


def make_face_tracker_key(bbox):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    w = x2 - x1
    h = y2 - y1
    return f"{round(cx/20)}_{round(cy/20)}_{round(w/20)}_{round(h/20)}"


def process_frame_face(frame):
    """
    Face recognition mode: Identify known faces or enroll new ones
    This is the full pipeline including face recognition
    """
    print("[PIPELINE] Running FACE RECOGNITION mode")
    
    cleanup_unknown_trackers()
    
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

            # Narrow the search to the upper portion of the person box so face_recognition
            # can find the face more reliably when YOLO returns a whole-person bounding box.
            h_crop, w_crop = crop.shape[:2]
            offset_x = int(w_crop * 0.1)
            face_roi = crop[0:max(1, int(h_crop * 0.5)), offset_x:max(offset_x + 1, int(w_crop * 0.9))]
            face_encs = []
            face_locs = []

            if face_roi.size > 0:
                rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_locs = face_recognition.face_locations(rgb_roi)
                face_encs = face_recognition.face_encodings(rgb_roi, face_locs)
                if face_locs:
                    face_locs = [
                        (top, right + offset_x, bottom, left + offset_x)
                        for (top, right, bottom, left) in face_locs
                    ]

            if not face_encs:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                face_locs = face_recognition.face_locations(rgb_crop)
                face_encs = face_recognition.face_encodings(rgb_crop, face_locs)

            if not face_encs:
                results.append({
                    "type": "face",
                    "name": "unknown",
                    "bbox": det["bbox"],
                    "announce": False,
                    "face_encoding": None
                })
                continue

            # Process ALL faces in this detection box (not just first)
            for face_enc in face_encs:
                best_name, best_dist = compare_face_to_db(face_enc, db)

                if best_name and best_dist < FACE_MATCH_THRESHOLD:
                    announce = update_presence(best_name)
                    names_this_frame.add(best_name)

                    results.append({
                        "type": "face",
                        "name": best_name,
                        "distance": best_dist,
                        "bbox": det["bbox"],
                        "announce": announce,
                        "face_encoding": None  # Don't transmit known faces
                    })
                    print(f"[FACE] Recognized: {best_name} (distance: {best_dist:.3f}, threshold: {FACE_MATCH_THRESHOLD})")
                else:
                    # Unknown face - include encoding for potential enrollment
                    bytes_data = face_enc.astype(np.float32).tobytes()
                    encoded_face = base64.b64encode(bytes_data).decode('utf-8')

                    tracker_key = make_face_tracker_key(det["bbox"])
                    now = time.time()
                    if tracker_key not in unknown_face_trackers:
                        unknown_face_trackers[tracker_key] = {
                            "first_seen": now,
                            "last_seen": now,
                            "bbox": det["bbox"]
                        }
                    else:
                        unknown_face_trackers[tracker_key]["last_seen"] = now
                        unknown_face_trackers[tracker_key]["bbox"] = det["bbox"]

                    seen_time = now - unknown_face_trackers[tracker_key]["first_seen"]
                    prompt_enroll = seen_time >= UNKNOWN_ENROLLMENT_DELAY

                    results.append({
                        "type": "face",
                        "name": "unknown",
                        "bbox": det["bbox"],
                        "announce": False,
                        "face_encoding": encoded_face,
                        "prompt_enrollment": prompt_enroll,
                        "unknown_seen_seconds": round(seen_time, 1)
                    })
                    print(f"[FACE] Unknown person (best match: {best_name}, dist: {best_dist:.3f}, threshold: {FACE_MATCH_THRESHOLD}, seen={seen_time:.1f}s)")
        else:
            results.append({
                "type": "object",
                "label": det["label"],
                "bbox": det["bbox"]
            })

    mark_absent(names_this_frame)
    print(f"[FACE RECOGNITION] Found {len(results)} detections")
    
    # Debug: Count unknown faces with encodings
    unknown_count = sum(1 for r in results if r.get('type') == 'face' and r.get('name') == 'unknown' and r.get('face_encoding'))
    if unknown_count > 0:
        print(f"[FACE RECOGNITION] Found {unknown_count} unknown face(s) with encodings for enrollment")
    
    return results


# Legacy function for backward compatibility
def process_frame(frame):
    """Backward compatible: defaults to face recognition mode"""
    return process_frame_face(frame)

