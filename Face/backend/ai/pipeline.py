# ai/pipeline.py

import cv2
import face_recognition
from ai.detector import detect_yolo, detect_faces_fallback
from ai.recognizer import load_database, compare_face_to_db
from ai.encounter import update_presence, mark_absent

FACE_MATCH_THRESHOLD = 0.6  # a bit more relaxed; typical values ~0.6
PERSON_LABELS = {"person", "face", "human"}

db = load_database()
print(f"[FACE DB] Loaded {len(db)} people: {list(db.keys())}")

def process_frame(frame):
    # Step 1: detect objects/persons
    detections = detect_yolo(frame)
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
                    "announce": False,
                    "enrollable": True    # <— ADD THIS
                })
                continue

            face_enc = face_encs[0]
            best_name, best_dist, best_idx = compare_face_to_db(face_enc, db)


            if best_name and best_dist < FACE_MATCH_THRESHOLD:
                announce = update_presence(best_name)
                names_this_frame.add(best_name)
                
                # strong match --> update adaptive encoding slot
                if best_dist < 0.45:  # strong confidence
                    from ai.recognizer import add_encoding_to_person, save_database
                    add_encoding_to_person(db, best_name, face_enc)
                    save_database(db)

                results.append({
                    "type": "face",
                    "name": best_name,
                    "distance": best_dist,
                    "bbox": det["bbox"],
                    "announce": announce
                })
                print({
                    "type": "face",
                    "name": best_name,
                    "distance": best_dist,
                    "bbox": det["bbox"],
                    "announce": announce
                })
            else:
                results.append({
                    "type": "face",
                    "name": "unknown",
                    "bbox": det["bbox"],
                    "announce": False
                })
                print(f"[FACE] Unknown / below threshold: best_name={best_name}, dist={best_dist:.3f}")
        else:
            results.append({
                "type": "object",
                "label": det["label"],
                "bbox": det["bbox"]
            })
            print({
                "type": "object",
                "label": det["label"],
                "bbox": det["bbox"]
            })

    mark_absent(names_this_frame)
    return results
