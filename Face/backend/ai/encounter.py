# ai/encounter.py

from time import time

RE_ENCOUNTER_TIME = 30.0  # seconds
encounter_state = {}

def update_presence(name):
    now = time()
    record = encounter_state.get(name)

    if record is None:
        encounter_state[name] = {
            "last_seen": now,
            "last_detected": now,
            "in_frame": True
        }
        print(f"[PRESENCE] New person: {name}, announcing")
        return True  # should announce

    last_detected = record["last_detected"]
    was_in_frame = record["in_frame"]
    elapsed_since_detected = now - last_detected

    if not was_in_frame:
        announce = elapsed_since_detected > RE_ENCOUNTER_TIME
        record["last_seen"] = now
        record["last_detected"] = now
        record["in_frame"] = True

        if announce:
            print(f"[PRESENCE] Re-encounter: {name} after {elapsed_since_detected:.1f}s since last detected (> {RE_ENCOUNTER_TIME}s), announcing")
        else:
            print(f"[PRESENCE] Re-encounter: {name} after {elapsed_since_detected:.1f}s since last detected (< {RE_ENCOUNTER_TIME}s), not announcing")
        return announce

    record["last_seen"] = now
    record["last_detected"] = now
    record["in_frame"] = True
    print(f"[PRESENCE] Continuing presence: {name}")
    return False


def mark_absent(names_in_frame):
    """Mark anyone not in this frame as absent."""
    absent_count = 0
    for name in encounter_state.keys():
        if name not in names_in_frame:
            encounter_state[name]["in_frame"] = False
            absent_count += 1
            print(f"[PRESENCE] Marked absent: {name}")
    if absent_count > 0:
        print(f"[PRESENCE] Marked {absent_count} people as absent")
