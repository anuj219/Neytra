# ai/pipeline.py

import cv2
import face_recognition
from ai.detector import detect_yolo, detect_faces_fallback
from ai.recognizer import load_database, compare_face_to_db
from ai.encounter import update_presence, mark_absent

FACE_MATCH_THRESHOLD = 0.5

db = load_database()

def process_frame(frame):
    # Step 1: detect objects/persons
    detections = detect_yolo(frame)
    if not detections:  # fallback
        detections = detect_faces_fallback(frame)

    names_this_frame = set()
    results = []

    for det in detections:
        label = det["label"].lower()
        x1, y1, x2, y2 = det["bbox"]

        if label == "person":
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_crop)
            face_encs = face_recognition.face_encodings(rgb_crop, face_locs)

            if not face_encs:
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
            else:
                # Include distance even if unknown (helps debug threshold issues)
                results.append({
                    "type": "face",
                    "name": "unknown",
                    "distance": best_dist if best_name else None,
                    "best_match": best_name if best_name else None,
                    "threshold": FACE_MATCH_THRESHOLD,
                    "bbox": det["bbox"],
                    "announce": False
                })

        else:
            results.append({
                "type": "object",
                "label": det["label"],
                "bbox": det["bbox"]
            })

    mark_absent(names_this_frame)
    return results
