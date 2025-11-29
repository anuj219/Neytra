
from ai.navigation import get_navigation_guidance

def test_navigation():
    width = 640
    
    # Scenario 1: Obstacle in Center, Left Clear
    detections_1 = [
        {"bbox": [250, 100, 390, 200], "label": "chair"} # Center is roughly 213-426
    ]
    guidance_1 = get_navigation_guidance(detections_1, width)
    print(f"Scenario 1 (Center Blocked, Left Clear): {guidance_1}")
    assert guidance_1 == "Obstacle ahead. Take a left."

    # Scenario 2: Obstacle in Center and Left, Right Clear
    detections_2 = [
        {"bbox": [250, 100, 390, 200], "label": "chair"}, # Center
        {"bbox": [50, 100, 150, 200], "label": "table"}  # Left
    ]
    guidance_2 = get_navigation_guidance(detections_2, width)
    print(f"Scenario 2 (Center & Left Blocked, Right Clear): {guidance_2}")
    assert guidance_2 == "Obstacle ahead. Take a right."

    # Scenario 3: No Obstacles
    detections_3 = []
    guidance_3 = get_navigation_guidance(detections_3, width)
    print(f"Scenario 3 (No Obstacles): {guidance_3}")
    assert guidance_3 is None

    # Scenario 4: Obstacle on Left only (Center Clear)
    detections_4 = [
        {"bbox": [50, 100, 150, 200], "label": "table"} # Left
    ]
    guidance_4 = get_navigation_guidance(detections_4, width)
    print(f"Scenario 4 (Left Blocked, Center Clear): {guidance_4}")
    assert guidance_4 is None

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_navigation()
