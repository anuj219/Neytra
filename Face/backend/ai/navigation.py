import time

NAVIGATION_COOLDOWN = 10.0  # seconds between navigation announcements
last_navigation_time = 0
last_navigation_text = ""  # track last guidance to avoid repeating identical ones

def get_navigation_guidance(detections, frame_width=640):
    """
    Analyze detections to provide navigation guidance.
    
    Args:
        detections: List of dictionaries with 'bbox' [x1, y1, x2, y2] and 'label'
        frame_width: Width of the frame (default 640)
        
    Returns:
        str: Navigation instruction or None if path is clear
    """
    global last_navigation_time, last_navigation_text
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
    obstacles_left = False
    obstacles_center = False
    obstacles_right = False
    
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
            
        x1, y1, x2, y2 = bbox
        label = det.get("label", "unknown")
        
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
        
    # Center is blocked - provide guidance
    # NOTE: Don't include person names here — face announcement handles that separately
    person_in_center = any(
        det.get("type") == "face"
        for det in detections
        if det.get("bbox") and det["bbox"][0] < right_boundary and det["bbox"][2] > left_boundary
    )
    
    if person_in_center:
        if not obstacles_left:
            guidance = "Person ahead. Move left."
        elif not obstacles_right:
            guidance = "Person ahead. Move right."
        else:
            guidance = "Person ahead. Path blocked."
    else:
        # Regular obstacle
        if not obstacles_left:
            guidance = "Obstacle ahead. Move left."
        elif not obstacles_right:
            guidance = "Obstacle ahead. Move right."
        else:
            guidance = "Path blocked. Please stop."
    
    # Don't repeat identical guidance
    if guidance == last_navigation_text and now - last_navigation_time < 30.0:
        print(f"[NAVIGATION] Same guidance as before, suppressing repeat")
        return None

    print(f"[NAVIGATION] Guidance: {guidance}")
    last_navigation_time = now
    last_navigation_text = guidance
    return guidance