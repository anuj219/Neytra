# ai/recognizer.py

import pickle
import os
import numpy as np
import face_recognition

# Prefer shared DB if available; fall back to backend-local DB
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Face/backend
FACE_DIR = os.path.dirname(BACKEND_DIR)  # Face
SHARED_DB = os.path.join(FACE_DIR, "faces.pkl")
BACKEND_DB = os.path.join(BACKEND_DIR, "db", "faces.pkl")

if os.path.exists(SHARED_DB):
    DB_PATH = SHARED_DB
else:
    DB_PATH = BACKEND_DB

def load_database():
    if not os.path.exists(DB_PATH):
        print(f"[FACE DB] No DB found at {DB_PATH}, starting empty.")
        return {}
    with open(DB_PATH, "rb") as f:
        data = pickle.load(f)

    # legacy format support
    if isinstance(data, dict) and "encodings" in data and "names" in data:
        encs, names = data["encodings"], data["names"]
        new = {}
        for enc, name in zip(encs, names):
            new.setdefault(name, []).append(enc)
        return new

    print(f"[FACE DB] Loaded DB from {DB_PATH} with {len(data)} people.")
    return data


def save_database(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

# 3 encodings per person, so we need to compare the face encoding to the 3 encodings and return the best match
def compare_face_to_db(face_encoding, db):
    best_name = None
    best_dist = 1.0
    best_idx = -1

    for name, enc_list in db.items():
        for idx, stored_enc in enumerate(enc_list):
            dist = np.linalg.norm(stored_enc - face_encoding)  # or face_recognition.distance()
            if dist < best_dist:
                best_dist = dist
                best_name = name
                best_idx = idx

    return best_name, best_dist, best_idx



# ----------- 3 Encoding structure ------------------

def add_encoding_to_person(db, name, new_enc):
    """
    Add a new encoding for a person using:
    [base, side, adaptive]
    Adaptive = last slot, overwritten as person re-appears.
    """
    if name not in db:
        db[name] = [new_enc]
        return
    
    enc_list = db[name]

    if len(enc_list) < 3:     # rn were keeping 3 as the max encoding/person
        enc_list.append(new_enc)
    else:
        # Overwrite adaptive slot (index 2)
        enc_list[-1] = new_enc
