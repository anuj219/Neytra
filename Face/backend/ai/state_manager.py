
import time

class StateManager:
    def __init__(self):
        # Configuration
        self.OBJECT_COOLDOWN = 5.0      # Seconds before announcing same object again
        self.NAVIGATION_COOLDOWN = 3.0  # Seconds before repeating same navigation instruction
        self.PERSON_RE_ENCOUNTER = 30.0 # Seconds before re-greeting a person
        
        # State
        self.last_announced_objects = {} # {label: last_time}
        self.last_navigation = {"instruction": None, "time": 0}
        self.person_state = {}           # {name: {"last_seen": time, "in_frame": bool}}
        
    def should_announce_object(self, label):
        """Check if we should announce this object based on cooldown."""
        now = time.time()
        last_time = self.last_announced_objects.get(label, 0)
        
        if now - last_time > self.OBJECT_COOLDOWN:
            self.last_announced_objects[label] = now
            return True
        return False
        
    def should_announce_navigation(self, instruction):
        """Check if we should announce this navigation instruction."""
        if not instruction:
            return False
            
        now = time.time()
        last_instr = self.last_navigation["instruction"]
        last_time = self.last_navigation["time"]
        
        # If instruction changed, announce immediately
        if instruction != last_instr:
            self.last_navigation = {"instruction": instruction, "time": now}
            return True
            
        # If same instruction, check cooldown
        if now - last_time > self.NAVIGATION_COOLDOWN:
            self.last_navigation["time"] = now
            return True
            
        return False

    def update_person_presence(self, name):
        """Track person presence and decide if we should greet."""
        now = time.time()
        
        if name not in self.person_state:
            self.person_state[name] = {"last_seen": now, "in_frame": True}
            return True # First time seeing them
            
        last_seen = self.person_state[name]["last_seen"]
        was_in_frame = self.person_state[name]["in_frame"]
        
        # Update current state
        self.person_state[name]["last_seen"] = now
        self.person_state[name]["in_frame"] = True
        
        # If they were gone for a while and came back
        if not was_in_frame and (now - last_seen > self.PERSON_RE_ENCOUNTER):
            return True
            
        return False
        
    def mark_absent(self, current_names):
        """Mark people not in current frame as absent."""
        for name in self.person_state:
            if name not in current_names:
                self.person_state[name]["in_frame"] = False

# Global instance
state_manager = StateManager()
