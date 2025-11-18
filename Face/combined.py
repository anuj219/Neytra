"""
Neytra_Integrated.py
Integrated YOLO (object detection) + face_recognition (ID) prototype.

Behavior:
 - YOLO runs continuously (if ultralytics available and you provide a .pt file).
 - For each detected "person" box, we crop the ROI and run face_recognition on the crop.
 - Recognized names + face boxes are drawn and spoken (threaded TTS).
 - Uses caching/cooldown per person location to avoid repeating heavy recognition each frame.

Config:
 - If you have a YOLO weights file (e.g. 'best.pt'), set MODEL_PATH below.
 - If ultralytics is not installed or MODEL_PATH is None, the script will fall back to using
   face_recognition on the whole frame at a lower frequency (for testing).
"""

from time import time
import time as _time
import cv2  
import face_recognition
import pickle
import os
import threading
import pyttsx3
import numpy as np


# ----- Name Listener (Press to speak) --------
import keyboard
import speech_recognition as sr

def listen_while_pressed(key, prompt_text=""):
    speak(prompt_text)
    print(f"Hold '{key.upper()}' and speak...")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio_data = None
        while True:
            if keyboard.is_pressed(key):  # user starts pressing key
                print("Listening...")
                audio_data = r.listen(source, phrase_time_limit=5)
            else:
                if audio_data:
                    break  # stop when key released
        try:
            text = r.recognize_google(audio_data).lower()
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            speak("Sorry, I didn’t catch that.")
            return None
        except Exception as e:
            print("Error in voice recognition:", e)
            return None

# ----- Optional YOLO import -----
YOLO_AVAILABLE = False
MODEL_PATH = 'C:\\Users\\anujv\\OneDrive\\Desktop\\Programming\\Codes\\python\\Face Recognition\\Neytra\\Object\\Neytra-Obj_Detection\\models\\bestv3.pt'  # if you know the model file path (eg. "best.pt"), put it here
 # if you know the model file path (eg. "best.pt"), put it here

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
    # it's fine — fallback will be used

# ---------- TTS helpers ----------
def speak(text):
    threading.Thread(target=_speak_thread, args=(text,), daemon=True).start()

def _speak_thread(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("TTS error:", e)

# ---------- DB helpers (supports old list format or dict format) ----------
DB_PATH = 'faces.pkl'

def load_database():
    """
    Supports two formats:
      1) {'encodings': [enc1, enc2,..], 'names': [n1, n2,..]}  (legacy)
      2) {'Anuj': [encA1, encA2], 'Rohit': [encR1, ...]}      (multi-enc per person)
    Returns a normalized dict -> {name: [encodings]}
    """
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, 'rb') as f:
        data = pickle.load(f)
    # legacy format
    if isinstance(data, dict) and 'encodings' in data and 'names' in data:
        encs = data['encodings']
        names = data['names']
        out = {}
        for e, n in zip(encs, names):
            out.setdefault(n, []).append(e)
        return out
    # assume already in desired format
    if isinstance(data, dict):
        # quick check: values are lists of arrays?
        return data
    # unexpected - return empty
    return {}

def save_database(db):
    """
    Saves in the new dict format: {name: [enc1, enc2, ...]}
    """
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)

# ---------- Utility functions ----------
def compare_face_to_db(face_encoding, db):
    """
    Compare a single face_encoding against db (dict name->list(encodings)).
    Returns (best_name, best_distance) or (None, None)
    """
    best_name = None
    best_dist = 1.0
    for name, enc_list in db.items():
        # compute distances against all stored encodings for that person
        distances = face_recognition.face_distance(enc_list, face_encoding)
        if len(distances) == 0:
            continue
        min_d = float(np.min(distances))
        if min_d < best_dist:
            best_dist = min_d
            best_name = name
    return best_name, best_dist

def center_of_box(box):
    """box = (x1,y1,x2,y2) -> center (cx, cy)"""
    x1,y1,x2,y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

def boxes_distance(b1, b2):
    c1 = center_of_box(b1); c2 = center_of_box(b2)
    return np.hypot(c1[0]-c2[0], c1[1]-c2[1])

# ---------- Load DB & model ----------
db = load_database()
print("Loaded DB names:", list(db.keys()))
speak(f"Loaded {len(db)} known people.")

yolo_model = None
if YOLO_AVAILABLE and MODEL_PATH:
    try:
        yolo_model = YOLO(MODEL_PATH)
        print("Loaded YOLO model:", MODEL_PATH)
    except Exception as e:
        print("Failed to load YOLO model:", e)
        yolo_model = None
else:
    if YOLO_AVAILABLE:
        print("Ultralytics installed but no MODEL_PATH provided. Using fallback person-only detection.")
    else:
        print("Ultralytics YOLO not available — falling back to face_recognition-only mode.")

# ---------- Runtime parameters (tune if needed) ----------
CONF_THRESHOLD = 0.5        # YOLO confidence threshold
FACE_MATCH_THRESHOLD = 0.50 # lower -> stricter; 0.5–0.55 recommended
PERSON_COOLDOWN = 3.0       # seconds to reuse last result for a nearby bbox
TTS_MIN_GAP = 1.0           # seconds between any two TTS calls
last_spoken = {}      # stores { phrase: last_time_spoken }

# ---------- ENCOUNTER STATE TRACKING ----------
RE_ENCOUNTER_TIME = 30.0  # seconds before re-greeting after person leaves
encounter_state = {}      # {name: {"last_seen": ts, "in_frame": bool}}

# Speech tracking (separate from encounter logic)
SPEAK_COOLDOWN = 2.0      # minimum gap between any two TTS calls
last_tts_time = 0.0



def announce_person(name, position):
    """
    Decides if we should speak a person's name based on encounter state.
    Returns True if we should speak, False otherwise.
    """
    now = time()
    
    # Case 1: Never seen before
    if name not in encounter_state:
        encounter_state[name] = {"last_seen": now, "in_frame": True}
        return True  # speak first time
    
    # Case 2: Seen before
    last_seen = encounter_state[name]["last_seen"]
    was_in_frame = encounter_state[name]["in_frame"]
    
    # Sub-case 2a: They were NOT in frame (absent), now returned
    if not was_in_frame:
        # Only re-greet if enough time passed
        if now - last_seen > RE_ENCOUNTER_TIME:
            encounter_state[name]["last_seen"] = now
            encounter_state[name]["in_frame"] = True
            return True  # speak after long absence
        else:
            # They returned but within 30s — don't speak yet
            encounter_state[name]["in_frame"] = True
            encounter_state[name]["last_seen"] = now
            return False
    
    # Sub-case 2b: They were already in frame (continuous presence)
    # Don't speak again, just update timestamp
    encounter_state[name]["last_seen"] = now
    encounter_state[name]["in_frame"] = True
    return False


def mark_person_absent(name):
    """Mark a person as absent (they left the frame)."""
    if name in encounter_state:
        encounter_state[name]["in_frame"] = False


def cleanup_encounter_state(max_age=120.0):
    """
    Optional: Remove very old people (> 2 minutes unseen) to save memory.
    Only call if you want to limit dict size.
    """
    now = time()
    names_to_remove = [
        name for name, state in encounter_state.items()
        if now - state["last_seen"] > max_age
    ]
    for name in names_to_remove:
        del encounter_state[name]


# ---------- Per-person cache to avoid repeating recognition each frame ----------
# Each entry: {id: {'box': (x1,y1,x2,y2), 'name': name_or_None, 'time': ts}}
recent_people = []



# Helper to update recent_people: match by spatial proximity
def find_recent_for_box(box, max_dist=80):
    for entry in recent_people:
        if boxes_distance(entry['box'], box) < max_dist:
            return entry
    return None

def update_recent(box, name):
    entry = find_recent_for_box(box)
    now = time()
    if entry:
        entry['box'] = box
        entry['name'] = name
        entry['time'] = now
    else:
        recent_people.append({'box': box, 'name': name, 'time': now})

def cleanup_recent(ttl=5.0):
    """
    Remove cached entries older than ttl (time-to-live) seconds.
    Keeps memory from growing unbounded.
    """
    global recent_people
    now = time()
    recent_people = [
        entry for entry in recent_people
        if now - entry['time'] < ttl
    ]


# ---------- Main loop ----------
cap = cv2.VideoCapture(0)
speak("Neytra activated.")
print("Starting camera... (press 'q' to quit)")

# If no YOLO available, we will fallback to scanning face_recognition on whole frame every K frames
fallback_frame_skip = 10
fallback_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    detections = []
    
    # --- Object detection step (if yolo available) ---
    if yolo_model is not None:
        try:
            results = yolo_model(frame, imgsz=320, verbose=False)  # smaller imgsz for speed
            r = results[0]
            # iterate boxes
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                    cls = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                    # safe mapping to name (if model.names exists)
                    cls_name = None
                    try:
                        cls_name = r.names[cls]
                    except Exception:
                        cls_name = str(cls)
                    # box.xyxy returns tensor; convert to ints
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    except Exception:
                        coords = box.xyxy.tolist()
                        if coords:
                            x1,y1,x2,y2 = map(int, coords[0])
                        else:
                            continue
                    if conf < CONF_THRESHOLD:
                        continue
                    detections.append({'label': cls_name, 'conf': conf, 'box': (x1,y1,x2,y2)})
        except Exception as e:
            # if YOLO call fails for some reason, fallback to face-only
            print("YOLO inference error:", e)
            yolo_model = None

    # --- Fallback: no YOLO available (or disabled) ---
    if yolo_model is None:
        # only run face detection every fallback_frame_skip frames to save CPU
        fallback_counter += 1
        if fallback_counter % fallback_frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb)
            for (top,right,bottom,left) in face_locs:
                # convert to x1,y1,x2,y2
                x1,y1,x2,y2 = left,top,right,bottom
                detections.append({'label': 'person', 'conf': 0.9, 'box': (x1,y1,x2,y2)})

    # ========== NEW: Track which people we detected THIS frame ==========
    people_detected_this_frame = set()  # names of people recognized this frame
    

    objects_to_speak = []

    # ========== PROCESS DETECTIONS ==========
    speech_queue = []  # collect what to speak (avoid duplicates)

    frame_center = frame_w // 2
    for det in detections:
        label = det['label'].lower()
        (x1, y1, x2, y2) = det['box']
        cx = (x1 + x2) // 2

        # Determine position
        if cx < frame_center - frame_w * 0.2:
            position = "on your left"
        elif cx > frame_center + frame_w * 0.2:
            position = "on your right"
        else:
            position = "ahead"

        # Draw detection box
        color = (255, 200, 0) if label == "person" else (255, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, det['label'], (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ← ADD THIS: Collect all objects to speak (not just persons)
        objects_to_speak.append(f"{det['label']} {position}")


        # --- If PERSON: run face recognition ---
        if label == "person":
            recent = find_recent_for_box((x1, y1, x2, y2), max_dist=80)
            
            # Try cached result first
            if recent and (time() - recent['time'] < PERSON_COOLDOWN) and recent.get('name'):
                name = recent['name']
                people_detected_this_frame.add(name)
                print(f"[FACE] Cached: {name}")
                
                # Draw face box
                fx1 = int(x1 + (x2 - x1) * 0.15)
                fy1 = int(y1 + (y2 - y1) * 0.05)
                fx2 = int(x2 - (x2 - x1) * 0.15)
                fy2 = int(y1 + (y2 - y1) * 0.45)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(frame, name, (fx1, fy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                continue

            # Run fresh face recognition
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_crop)
            face_encs = face_recognition.face_encodings(rgb_crop, face_locs)

            if not face_encs:
                print("[FACE] No face in person region")
                update_recent((x1, y1, x2, y2), None)
                continue

            face_loc = face_locs[0]
            top, right, bottom, left = face_loc
            fx1 = x1 + left
            fy1 = y1 + top
            fx2 = x1 + right
            fy2 = y1 + bottom

            face_encoding = face_encs[0]
            best_name, best_dist = compare_face_to_db(face_encoding, db)

            # --- KNOWN PERSON ---
            if best_name is not None and best_dist < FACE_MATCH_THRESHOLD:
                name = best_name
                people_detected_this_frame.add(name)
                print(f"[FACE] Recognized: {name} (distance {best_dist:.2f})")
                update_recent((x1, y1, x2, y2), name)
                
                # Draw green box for known person
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(frame, name, (fx1, fy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # ========== USE ENCOUNTER LOGIC ==========
                print("\n")
                if announce_person(name, position):
                    print("30 Second Cooldown acheived")
                    speech_queue.append(f"{name} detected {position}")
                else:
                    print("Failed to cross 30 second cooldown")
                print(f" {name}'s (time_last_seen {encounter_state[name]['last_seen']})")
                print("\n")
                
            # --- UNKNOWN PERSON ---
            else:
                print("[FACE] Unknown person detected")
                speak("Unknown person detected. Press V and say Hello to add them, or P to cancel")

                adding_key = keyboard.read_key().lower()
                if adding_key == 'p':
                    continue  # skip enrollment

                response = listen_while_pressed('v')
                if not response or "hello" not in response:
                    speak("Okay, not adding new person.")
                    continue

                # Get name
                name = None
                while not name:
                    name = listen_while_pressed('v', "Press V and say the name of this person.")
                    if not name:
                        continue

                    speak(f"You said {name}. Press C to confirm or P key to cancel.")
                    confirm_key = keyboard.read_key().lower()

                    if confirm_key == 'c':
                        db.setdefault(name, []).append(face_encoding)
                        save_database(db)
                        speak(f"{name} added to memory.")
                        print(f"{name} added to DB.")
                        
                        # Mark as seen so we don't repeat name immediately
                        encounter_state[name] = {"last_seen": time(), "in_frame": True}
                        people_detected_this_frame.add(name)
                    else:
                        speak("Cancelled.")
                    break
        
    # ========== MARK ABSENT: Anyone not detected this frame ==========
    for name in encounter_state.keys():
        if name not in people_detected_this_frame:
            mark_person_absent(name)

    # ========== SPEAK QUEUED MESSAGES ==========
    cleanup_recent(ttl=5.0)
    
    # Combine both object and face outputs (avoid duplicates)
    to_say = list(set(objects_to_speak + speech_queue))
    # current_time = time()
    # final_to_say = []

    
    current_time = time()
    if to_say and (current_time - last_tts_time > SPEAK_COOLDOWN):
        sentence = ", ".join(to_say)
        print(f"[SPEECH] {sentence}")
        speak(sentence)
        last_tts_time = current_time

    # Show frame
    cv2.imshow('Neytra Integrated', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
speak("Neytra shutting down.")


