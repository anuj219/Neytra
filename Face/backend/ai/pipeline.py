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
# UNKNOWN_ENROLLMENT_DELAY = 10.0  # seconds a face must remain in frame before prompting enrollment
DETECTION_COUNT_THRESHOLD = 3  # Number of detections needed to trigger enrollment
DETECTION_WINDOW_SECONDS = 15.0  # Time window in seconds for detection count
UNKNOWN_TRACKER_TTL = 5.0
UNKNOWN_TRACKER_MATCH_DIST = 0.6  # Encoding distance threshold for grouping the same unknown face
UNKNOWN_TRACKER_MAX_CENTER_DIST = 150   # Pixel distance threshold for bounding box center movement
UNKNOWN_TRACKER_MAX_ENCODINGS = 5  # Keep recent encodings for more stable matching
unknown_face_trackers = {}
unknown_tracker_id = 0
# Per-person cache to avoid repeating recognition each frame
recent_people = []
PERSON_COOLDOWN = 3.0  # seconds to reuse last result for a nearby bbox
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
    cutoff_time = now - DETECTION_WINDOW_SECONDS - 5.0  # Keep up to 20s worth of data
    stale_keys = []
    before_count = len(unknown_face_trackers)
    print(f"[TRACKER CLEANUP] Starting cleanup, {before_count} trackers")
    for key, tracker in unknown_face_trackers.items():
        before_ts = len(tracker["timestamps"])
        tracker["timestamps"] = [t for t in tracker["timestamps"] if t > cutoff_time]
        after_ts = len(tracker["timestamps"])
        if before_ts != after_ts:
            print(f"[TRACKER CLEANUP] Tracker {key}: removed {before_ts - after_ts} old timestamps")
        if not tracker["timestamps"]:
            stale_keys.append(key)
            print(f"[TRACKER CLEANUP] Marking tracker {key} as stale (no timestamps)")
    for key in stale_keys:
        del unknown_face_trackers[key]
    after_count = len(unknown_face_trackers)
    if before_count != after_count:
        print(f"[TRACKER CLEANUP] Removed {before_count - after_count} stale trackers, {after_count} remaining")


def get_position_from_bbox(bbox, frame_width):
    """Determine left/center/right position from bounding box"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    left_boundary = frame_width * 0.33
    right_boundary = frame_width * 0.66
    
    if center_x < left_boundary:
        return "left"
    elif center_x > right_boundary:
        return "right"
    else:
        return "center"


def find_matching_unknown_tracker(face_enc, bbox):
    best_id = None
    best_dist = float("inf")
    print(f"[TRACKER] Looking for matching tracker for bbox {bbox}")

    for tracker_id, tracker in unknown_face_trackers.items():
        if "encodings" not in tracker or not tracker["encodings"]:
            continue

        center_dist = tracker_center_distance(bbox, tracker["bbox"])
        print(f"[TRACKER] Checking tracker {tracker_id}: center_dist={center_dist:.1f}, max={UNKNOWN_TRACKER_MAX_CENTER_DIST}")
        if center_dist > UNKNOWN_TRACKER_MAX_CENTER_DIST:
            continue

        # Compare against the tracker's recent encodings
        for old_enc in tracker["encodings"]:
            from face_recognition import face_distance
            dist = float(face_distance([old_enc], face_enc)[0])
            print(f"[TRACKER] Distance to tracker {tracker_id}: {dist:.4f}")
            if dist < best_dist:
                best_dist = dist
                best_id = tracker_id

    if best_dist <= UNKNOWN_TRACKER_MATCH_DIST:
        print(f"[TRACKER] Matched tracker {best_id} with dist {best_dist:.4f} <= {UNKNOWN_TRACKER_MATCH_DIST}")
        return best_id
    else:
        print(f"[TRACKER] No match found, best dist {best_dist:.4f} > {UNKNOWN_TRACKER_MATCH_DIST}")
        return None


def tracker_center_distance(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    x3, y3, x4, y4 = bbox2
    cx2 = (x3 + x4) / 2
    cy2 = (y3 + y4) / 2
    return np.hypot(cx1 - cx2, cy1 - cy2)


# Helper functions for per-person caching
def center_of_box(box):
    """box = (x1,y1,x2,y2) -> center (cx, cy)"""
    x1,y1,x2,y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

def boxes_distance(b1, b2):
    c1 = center_of_box(b1); c2 = center_of_box(b2)
    return np.hypot(c1[0]-c2[0], c1[1]-c2[1])

def find_recent_for_box(box, max_dist=80):
    """Find cached entry for optimization only. Don't use for announcement logic."""
    now = time.time()
    print(f"[CACHE] Looking for recent entry for box {box}, max_dist={max_dist}")
    for entry in recent_people:
        if now - entry['time'] > PERSON_COOLDOWN:
            print(f"[CACHE] Skipping expired entry: {entry['name']} at {now - entry['time']:.1f}s ago")
            continue
        dist = boxes_distance(entry['box'], box)
        print(f"[CACHE] Checking entry {entry['name']}: dist={dist:.1f}, time_diff={now - entry['time']:.1f}s")
        if dist < max_dist:
            print(f"[CACHE] Found matching recent entry: {entry['name']} (dist={dist:.1f})")
            return entry
    print(f"[CACHE] No recent entry found for box {box}")
    return None

def update_recent(box, name):
    now = time.time()
    entry = find_recent_for_box(box)
    if entry:
        print(f"[CACHE] Updating existing entry: {entry['name']} -> {name}")
        entry['box'] = box
        entry['name'] = name
        entry['time'] = now
    else:
        print(f"[CACHE] Adding new entry: {name} at {box}")
        recent_people.append({'box': box, 'name': name, 'time': now})

def cleanup_recent(ttl=5.0):
    """
    Remove cached entries older than ttl seconds.
    """
    global recent_people
    now = time.time()
    before = len(recent_people)
    recent_people = [
        entry for entry in recent_people
        if now - entry['time'] < ttl
    ]
    after = len(recent_people)
    if before != after:
        print(f"[CACHE] Cleaned up {before - after} expired entries, {after} remaining")


def process_frame_face(frame):
    """
    Face recognition mode: Identify known faces or enroll new ones
    This is the full pipeline including face recognition
    """
    print("[PIPELINE] Running FACE RECOGNITION mode")
    
    cleanup_unknown_trackers()
    
    frame_width = frame.shape[1]
    
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

    print(f"[PIPELINE] Processing {len(detections)} detections")

    for det in detections:
        label = det["label"].lower()
        x1, y1, x2, y2 = det["bbox"]
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        
        if x2 <= x1 or y2 <= y1:
            print(f"[FACE] Skipping invalid bbox: {x1},{y1},{x2},{y2}")
            continue

        print(f"[FACE] Processing detection: {label} at {x1},{y1},{x2},{y2}")

        if label in PERSON_LABELS:
            # Check cache first
            recent = find_recent_for_box((x1, y1, x2, y2), max_dist=80)
            now = time.time()
            
            if recent and (now - recent['time'] < PERSON_COOLDOWN) and recent.get('name'):
                name = recent['name']
                names_this_frame.add(name)
                announce = update_presence(name)
                
                results.append({
                    "type": "face",
                    "name": name,
                    "bbox": det["bbox"],
                    "position": get_position_from_bbox(det["bbox"], frame_width),
                    "announce": announce,
                    "face_encoding": None
                })
                print(f"[FACE] Cached: {name}")
                continue

            # Run fresh face recognition
            print(f"[FACE] Running fresh recognition for {label}")
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"[FACE] Empty crop for {label}")
                update_recent((x1, y1, x2, y2), None)
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
                print(f"[FACE] Found {len(face_encs)} faces in ROI")

            if not face_encs:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                face_locs = face_recognition.face_locations(rgb_crop)
                face_encs = face_recognition.face_encodings(rgb_crop, face_locs)
                print(f"[FACE] Found {len(face_encs)} faces in full crop")

            if not face_encs:
                print(f"[FACE] No faces found in {label} detection")
                update_recent((x1, y1, x2, y2), None)
                results.append({
                    "type": "face",
                    "name": "unknown",
                    "bbox": det["bbox"],
                    "position": get_position_from_bbox(det["bbox"], frame_width),
                    "announce": False,
                    "face_encoding": None
                })
                continue

            # Process ALL faces in this detection box (not just first)
            for face_enc in face_encs:
                best_name, best_dist = compare_face_to_db(face_enc, db)

                if best_name and best_dist < FACE_MATCH_THRESHOLD:
                    announce = update_presence(best_name)  # This checks 30-second RE_ENCOUNTER_TIME
                    names_this_frame.add(best_name)
                    update_recent((x1, y1, x2, y2), best_name)

                    results.append({
                        "type": "face",
                        "name": best_name,
                        "distance": best_dist,
                        "bbox": det["bbox"],
                        "position": get_position_from_bbox(det["bbox"], frame_width),
                        "announce": announce,
                        "face_encoding": None,  # Don't transmit known faces
                        "announcement_reason": "30-second cooldown check via encounter_state"
                    })
                    print(f"[FACE] Recognized: {best_name} (distance: {best_dist:.3f}, threshold: {FACE_MATCH_THRESHOLD}, announce={announce})")
                else:
                    # Unknown face - include encoding for potential enrollment
                    bytes_data = face_enc.astype(np.float32).tobytes()
                    encoded_face = base64.b64encode(bytes_data).decode('utf-8')

                    now = time.time()
                    match_id = find_matching_unknown_tracker(face_enc, det["bbox"])

                    if match_id is not None:
                        tracker = unknown_face_trackers[match_id]
                        tracker["timestamps"].append(now)
                        tracker["bbox"] = det["bbox"]
                        tracker["last_encoding"] = face_enc
                        tracker["last_seen"] = now
                        tracker["encodings"].append(face_enc)
                        tracker["encodings"] = tracker["encodings"][-UNKNOWN_TRACKER_MAX_ENCODINGS:]
                        tracker_id = match_id
                        matched = True
                    else:
                        global unknown_tracker_id
                        unknown_tracker_id += 1
                        tracker_id = f"unknown_{unknown_tracker_id}"
                        unknown_face_trackers[tracker_id] = {
                            "timestamps": [now],
                            "last_seen": now,
                            "bbox": det["bbox"],
                            "last_encoding": face_enc,
                            "encodings": [face_enc]
                        }
                        matched = False

                    recent_detections = [t for t in unknown_face_trackers[tracker_id]["timestamps"] if now - t <= DETECTION_WINDOW_SECONDS]
                    unknown_face_trackers[tracker_id]["timestamps"] = recent_detections
                    detection_count = len(recent_detections)
                    prompt_enroll = detection_count >= DETECTION_COUNT_THRESHOLD
                    seen_time = now - min(recent_detections) if recent_detections else 0.0

                    results.append({
                        "type": "face",
                        "name": "unknown",
                        "bbox": det["bbox"],
                        "position": get_position_from_bbox(det["bbox"], frame_width),
                        "announce": False,
                        "face_encoding": encoded_face,
                        "prompt_enrollment": prompt_enroll,
                        "unknown_seen_seconds": round(seen_time, 1),
                        "detection_count": detection_count,
                        "tracker_id": tracker_id,
                        "matched_existing_tracker": matched
                    })
                    print(f"[FACE] Unknown person (tracker={tracker_id}, matched={matched}, best match: {best_name}, dist: {best_dist:.3f}, threshold: {FACE_MATCH_THRESHOLD}, detections={detection_count}, seen={seen_time:.1f}s, prompt_enroll={prompt_enroll})")
                    # Don't cache unknown faces to allow tracking
                    if prompt_enroll:
                        print(f"[ENROLL TRIGGER] Tracker {tracker_id} met threshold with {detection_count} detections in {DETECTION_WINDOW_SECONDS}s")
        else:
            results.append({
                "type": "object",
                "label": det["label"],
                "bbox": det["bbox"]
            })

    mark_absent(names_this_frame)
    cleanup_recent()
    cleanup_unknown_trackers()
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

