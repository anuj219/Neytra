# ai/recognizer.py

import pickle
import os
import numpy as np
import face_recognition

# Get absolute path to db directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "faces.pkl")

def load_database():
    if not os.path.exists(DB_PATH):
        print(f"Warning: Database not found at {DB_PATH}")
        return {}
    with open(DB_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded database from {DB_PATH} with {len(data)} entries")

    # legacy format support
    if isinstance(data, dict) and "encodings" in data and "names" in data:
        encs, names = data["encodings"], data["names"]
        new = {}
        for enc, name in zip(encs, names):
            new.setdefault(name, []).append(enc)
        return new

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
