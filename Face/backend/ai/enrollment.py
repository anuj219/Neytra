# ai/enrollment.py
"""
Unknown Person Enrollment Module
Handles the workflow for enrolling new faces when an unknown person is detected.

Workflow:
1. Unknown person detected (low confidence or not in DB)
2. Ask user via voice: "Would you like to add this person?"
3. If yes, ask for name via voice
4. Confirm the name
5. Save to database
"""

import base64
import numpy as np
import face_recognition
from ai.recognizer import save_database, load_database, compare_face_to_db

# Enrollment state tracking
enrollment_state = {
    "pending_encoding": None,
    "pending_name": None,
    "in_enrollment": False
}


def initiate_enrollment(face_encoding):
    """
    Start enrollment process for an unknown person.
    Store the face encoding temporarily.
    
    Args:
        face_encoding: numpy array of face encoding
        
    Returns:
        dict: {"status": "initiated", "message": "Ask user to confirm"}
    """
    if enrollment_state["in_enrollment"]:
        return {"status": "busy", "message": "Already in enrollment process"}
    
    enrollment_state["pending_encoding"] = face_encoding
    enrollment_state["in_enrollment"] = True
    
    return {
        "status": "initiated",
        "message": "Unknown person detected. Do you want to add them? Say yes or no.",
        "action": "ask_confirmation"
    }


def ask_for_name():
    """
    Request name from user during enrollment.
    
    Returns:
        dict: {"status": "awaiting_name", "action": "capture_name"}
    """
    if not enrollment_state["in_enrollment"]:
        return {"status": "error", "message": "Not in enrollment process"}
    
    return {
        "status": "awaiting_name",
        "message": "Press and hold your microphone, then say the person's name clearly.",
        "action": "capture_name_voice"
    }


def confirm_name(name):
    """
    Confirm the captured name and prepare to save.
    
    Args:
        name: Captured name from voice
        
    Returns:
        dict: {"status": "awaiting_confirmation", "action": "confirm_voice"}
    """
    if not enrollment_state["in_enrollment"]:
        return {"status": "error", "message": "Not in enrollment process"}
    
    enrollment_state["pending_name"] = name
    
    return {
        "status": "awaiting_confirmation",
        "name": name,
        "message": f"You said {name}. Is this correct? Say yes or no.",
        "action": "confirm_name_voice"
    }


def complete_enrollment(confirmed=True):
    """
    Complete the enrollment process and save to database.
    
    Args:
        confirmed (bool): Whether user confirmed the name
        
    Returns:
        dict: Success or error status
    """
    if not enrollment_state["in_enrollment"]:
        return {"status": "error", "message": "Not in enrollment process"}
    
    if not confirmed:
        cancel_enrollment()
        return {"status": "cancelled", "message": "Enrollment cancelled."}
    
    # Validate we have both encoding and name
    if enrollment_state["pending_encoding"] is None:
        return {"status": "error", "message": "No face encoding available"}
    
    if enrollment_state["pending_name"] is None:
        return {"status": "error", "message": "No name provided"}
    
    name = enrollment_state["pending_name"]
    face_enc = enrollment_state["pending_encoding"]
    
    # Load database
    db = load_database()
    
    # Add face to database
    db.setdefault(name, []).append(face_enc)
    save_database(db)
    
    # Reset state
    cancel_enrollment()
    
    return {
        "status": "success",
        "message": f"{name} has been added to memory.",
        "name": name
    }


def cancel_enrollment():
    """Cancel the current enrollment process."""
    enrollment_state["pending_encoding"] = None
    enrollment_state["pending_name"] = None
    enrollment_state["in_enrollment"] = False


def encode_face_for_transmission(face_encoding):
    """
    Convert face encoding to base64 for transmission over HTTP.
    
    Args:
        face_encoding: numpy array
        
    Returns:
        str: base64 encoded string
    """
    bytes_data = face_encoding.astype(np.float32).tobytes()
    return base64.b64encode(bytes_data).decode('utf-8')


def decode_face_from_transmission(encoded_str):
    """
    Decode face encoding from base64.
    
    Args:
        encoded_str: base64 encoded string
        
    Returns:
        numpy array: face encoding
    """
    bytes_data = base64.b64decode(encoded_str.encode('utf-8'))
    return np.frombuffer(bytes_data, dtype=np.float32)


def save_unknown_face(face_encoding, name):
    """
    Directly save an unknown face with a provided name.
    Used as a shortcut when you already have both encoding and name.
    
    Args:
        face_encoding: numpy array of face encoding
        name: Person's name
        
    Returns:
        dict: Success or error
    """
    try:
        db = load_database()
        db.setdefault(name, []).append(face_encoding)
        save_database(db)
        
        return {
            "status": "success",
            "message": f"{name} added to database",
            "name": name
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save: {str(e)}"
        }
