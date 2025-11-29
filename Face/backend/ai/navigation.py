
def get_navigation_guidance(detections, frame_width=640):
    """
    Analyze detections to provide navigation guidance.
    
    Args:
        detections: List of dictionaries with 'bbox' [x1, y1, x2, y2] and 'label'
        frame_width: Width of the frame (default 640)
        
    Returns:
        str: Navigation instruction or None if path is clear
    """
    if not detections:
        return None

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
        
        # Check for overlap with zones
        # Center Zone: [left_boundary, right_boundary]
        # Object X range: [x1, x2]
        
        # Check if object overlaps with Center Zone
        if x1 < right_boundary and x2 > left_boundary:
            obstacles_center = True
            
        # Check if object overlaps with Left Zone (0 to left_boundary)
        if x1 < left_boundary:
            obstacles_left = True
            
        # Check if object overlaps with Right Zone (right_boundary to width)
        if x2 > right_boundary:
            obstacles_right = True
            
    # Logic for guidance
    if not obstacles_center:
        # Path ahead is clear
        return None
        
    # Center is blocked
    if not obstacles_left:
        return "Obstacle ahead. Take a left."
    elif not obstacles_right:
        return "Obstacle ahead. Take a right."
    else:
        # Both left and right are blocked
        return "Path blocked. Please stop."
