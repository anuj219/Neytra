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


def compare_face_to_db(face_encoding, db):
    best_name, best_dist = None, 1.0
    for name, enc_list in db.items():
        if not enc_list:
            continue
        distances = face_recognition.face_distance(enc_list, face_encoding)
        min_d = float(np.min(distances))
        if min_d < best_dist:
            best_dist, best_name = min_d, name
    return best_name, best_dist
