# ai/encounter.py

from time import time

RE_ENCOUNTER_TIME = 30.0  # seconds – only re-announce after this gap
encounter_state = {}

def update_presence(name):
    """
    Track a known person's presence and decide whether to announce them aloud.

    Rules:
      1. First time ever seeing this person → announce.
      2. Every detection updates `last_detected` (used by frontend for display).
      3. `last_announced` is ONLY updated when we actually announce.
      4. Announce aloud only if (now - last_announced) >= RE_ENCOUNTER_TIME.
      5. This is per-person.
    """
    now = time()
    record = encounter_state.get(name)

    if record is None:
        # Brand-new person – announce immediately
        encounter_state[name] = {
            "last_seen": now,
            "last_detected": now,
            "last_announced": now,
            "in_frame": True
        }
        print(f"[PRESENCE] New person: {name}, announcing")
        return True  # should announce

    # Always update detection timestamp & visibility
    record["last_seen"] = now
    record["last_detected"] = now
    record["in_frame"] = True

    # Decide whether to announce based on time since LAST ANNOUNCEMENT
    elapsed_since_announced = now - record["last_announced"]

    if elapsed_since_announced >= RE_ENCOUNTER_TIME:
        record["last_announced"] = now
        print(f"[PRESENCE] Re-announce: {name} after {elapsed_since_announced:.1f}s since last announced (>= {RE_ENCOUNTER_TIME}s)")
        return True

    print(f"[PRESENCE] Suppressed: {name} (only {elapsed_since_announced:.1f}s since last announced, need {RE_ENCOUNTER_TIME}s)")
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