import time

NAVIGATION_COOLDOWN = 5.0  # seconds between announcements
last_navigation_time = 0

def get_navigation_guidance(detections, frame_width=640):
    """
    Analyze detections to provide navigation guidance.
    
    Args:
        detections: List of dictionaries with 'bbox' [x1, y1, x2, y2] and 'label'
        frame_width: Width of the frame (default 640)
        
    Returns:
        str: Navigation instruction or None if path is clear
    """
    global last_navigation_time
    now = time.time()
    if now - last_navigation_time < NAVIGATION_COOLDOWN:
        print(f"[NAVIGATION] Cooldown active ({now - last_navigation_time:.1f}s < {NAVIGATION_COOLDOWN}s), skipping announcement")
        return None
    
    if not detections:
        print("[NAVIGATION] No detections, path clear")
        return None

    print(f"[NAVIGATION] Analyzing {len(detections)} detections for guidance")

    # Define zones
    # Left: 0 to 33%
    # Center: 33% to 66%
    # Right: 66% to 100%
    
    left_boundary = frame_width * 0.33
    right_boundary = frame_width * 0.66
    
    # Check for obstacles in each zone
    # We consider an object an obstacle if it's in the zone
    # We can refine this by checking if it's "close" (large bbox area) or specific types
    # For now, assume all detected objects are potential obstacles as per user request
    
    obstacles_left = False
    obstacles_center = False
    obstacles_right = False
    
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
            
        x1, y1, x2, y2 = bbox
        label = det.get("label", "unknown")
        
        # Check for overlap with zones
        # Center Zone: [left_boundary, right_boundary]
        # Object X range: [x1, x2]
        
        # Check if object overlaps with Center Zone
        if x1 < right_boundary and x2 > left_boundary:
            obstacles_center = True
            print(f"[NAVIGATION] Obstacle in center: {label} at x1={x1:.0f}, x2={x2:.0f}")
            
        # Check if object overlaps with Left Zone (0 to left_boundary)
        if x1 < left_boundary:
            obstacles_left = True
            print(f"[NAVIGATION] Obstacle in left: {label} at x1={x1:.0f}, x2={x2:.0f}")
            
        # Check if object overlaps with Right Zone (right_boundary to width)
        if x2 > right_boundary:
            obstacles_right = True
            print(f"[NAVIGATION] Obstacle in right: {label} at x1={x1:.0f}, x2={x2:.0f}")
            
    print(f"[NAVIGATION] Zone status: left={obstacles_left}, center={obstacles_center}, right={obstacles_right}")
    
    # Logic for guidance
    if not obstacles_center:
        # Path ahead is clear
        print("[NAVIGATION] Path clear")
        return None
        
    # Center is blocked - provide more specific guidance
    person_in_center = any(det.get("type") == "face" and det.get("name") != "unknown" for det in detections if det.get("bbox") and det["bbox"][0] < right_boundary and det["bbox"][2] > left_boundary)
    unknown_person_in_center = any(det.get("type") == "face" and det.get("name") == "unknown" for det in detections if det.get("bbox") and det["bbox"][0] < right_boundary and det["bbox"][2] > left_boundary)
    
    if person_in_center:
        known_names = [det.get("name") for det in detections if det.get("type") == "face" and det.get("name") != "unknown" and det.get("bbox") and det["bbox"][0] < right_boundary and det["bbox"][2] > left_boundary]
        if not obstacles_left:
            guidance = f"Person ahead ({', '.join(known_names)}). Take a left."
        elif not obstacles_right:
            guidance = f"Person ahead ({', '.join(known_names)}). Take a right."
        else:
            guidance = f"Person ahead ({', '.join(known_names)}). Path blocked."
    elif unknown_person_in_center:
        if not obstacles_left:
            guidance = "Unknown person ahead. Take a left."
        elif not obstacles_right:
            guidance = "Unknown person ahead. Take a right."
        else:
            guidance = "Unknown person ahead. Path blocked."
    else:
        # Regular obstacle
        if not obstacles_left:
            guidance = "Obstacle ahead. Take a left."
        elif not obstacles_right:
            guidance = "Obstacle ahead. Take a right."
        else:
            guidance = "Path blocked. Please stop."
    
    print(f"[NAVIGATION] Guidance: {guidance}")
    last_navigation_time = now
    return guidance
def get_navigation_guidance(detections, frame_width=640):
    """
    Analyze detections to provide navigation guidance.
    
    Args:
        detections: List of dictionaries with 'bbox' [x1, y1, x2, y2] and 'label'
        frame_width: Width of the frame (default 640)
        
    Returns:
        str: Navigation instruction or None if path is clear
    """
    global last_navigation_time
    now = time.time()
    if now - last_navigation_time < NAVIGATION_COOLDOWN:
        print(f"[NAVIGATION] Cooldown active ({now - last_navigation_time:.1f}s < {NAVIGATION_COOLDOWN}s), skipping announcement")
        return None
    
    if not detections:
        print("[NAVIGATION] No detections, path clear")
        return None

    print(f"[NAVIGATION] Analyzing {len(detections)} detections for guidance")

    # Define zones
    # Left: 0 to 33%
    # Center: 33% to 66%
    # Right: 66% to 100%
    
    left_boundary = frame_width * 0.33
    right_boundary = frame_width * 0.66
    
    # Check for obstacles in each zone
    # We consider an object an obstacle if it's in the zone
    # We can refine this by checking if it's "close" (large bbox area) or specific types
    # For now, assume all detected objects are potential obstacles as per user request
    
    obstacles_left = False
    obstacles_center = False
    obstacles_right = False
    
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
            
        x1, y1, x2, y2 = bbox
        label = det.get("label", "unknown")
        
        # Check for overlap with zones
        # Center Zone: [left_boundary, right_boundary]
        # Object X range: [x1, x2]
        
        # Check if object overlaps with Center Zone
        if x1 < right_boundary and x2 > left_boundary:
            obstacles_center = True
            print(f"[NAVIGATION] Obstacle in center: {label} at x1={x1:.0f}, x2={x2:.0f}")
            
        # Check if object overlaps with Left Zone (0 to left_boundary)
        if x1 < left_boundary:
            obstacles_left = True
            print(f"[NAVIGATION] Obstacle in left: {label} at x1={x1:.0f}, x2={x2:.0f}")
            
        # Check if object overlaps with Right Zone (right_boundary to width)
        if x2 > right_boundary:
            obstacles_right = True
            print(f"[NAVIGATION] Obstacle in right: {label} at x1={x1:.0f}, x2={x2:.0f}")
            
    print(f"[NAVIGATION] Zone status: left={obstacles_left}, center={obstacles_center}, right={obstacles_right}")
    
    # Logic for guidance
    if not obstacles_center:
        # Path ahead is clear
        print("[NAVIGATION] Path clear")
        return None
        
    # Center is blocked - provide more specific guidance
    person_in_center = any(det.get("type") == "face" and det.get("name") != "unknown" for det in detections if det.get("bbox") and det["bbox"][0] < right_boundary and det["bbox"][2] > left_boundary)
    unknown_person_in_center = any(det.get("type") == "face" and det.get("name") == "unknown" for det in detections if det.get("bbox") and det["bbox"][0] < right_boundary and det["bbox"][2] > left_boundary)
    
    if person_in_center:
        known_names = [det.get("name") for det in detections if det.get("type") == "face" and det.get("name") != "unknown" and det.get("bbox") and det["bbox"][0] < right_boundary and det["bbox"][2] > left_boundary]
        if not obstacles_left:
            guidance = f"Person ahead ({', '.join(known_names)}). Take a left."
        elif not obstacles_right:
            guidance = f"Person ahead ({', '.join(known_names)}). Take a right."
        else:
            guidance = f"Person ahead ({', '.join(known_names)}). Path blocked."
    elif unknown_person_in_center:
        if not obstacles_left:
            guidance = "Unknown person ahead. Take a left."
        elif not obstacles_right:
            guidance = "Unknown person ahead. Take a right."
        else:
            guidance = "Unknown person ahead. Path blocked."
    else:
        # Regular obstacle
        if not obstacles_left:
            guidance = "Obstacle ahead. Take a left."
        elif not obstacles_right:
            guidance = "Obstacle ahead. Take a right."
        else:
            guidance = "Path blocked. Please stop."
    
    print(f"[NAVIGATION] Guidance: {guidance}")
    last_navigation_time = now
    return guidance
