# ai/encounter.py

from time import time

RE_ENCOUNTER_TIME = 30.0  # seconds
encounter_state = {}

def update_presence(name):
    now = time()
    if name not in encounter_state:
        encounter_state[name] = {"last_seen": now, "in_frame": True}
        return True  # should announce

    last_seen = encounter_state[name]["last_seen"]
    was_in_frame = encounter_state[name]["in_frame"]

    if not was_in_frame:
        if now - last_seen > RE_ENCOUNTER_TIME:
            encounter_state[name] = {"last_seen": now, "in_frame": True}
            return True
        else:
            encounter_state[name] = {"last_seen": now, "in_frame": True}
            return False

    encounter_state[name]["last_seen"] = now
    encounter_state[name]["in_frame"] = True
    return False


def mark_absent(names_in_frame):
    """Mark anyone not in this frame as absent."""
    for name in encounter_state.keys():
        if name not in names_in_frame:
            encounter_state[name]["in_frame"] = False
