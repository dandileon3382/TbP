
import json
import os
import sys
import math

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from website.backend.exercise_logic import ExerciseState

def test_squat_logic():
    print("\n--- Testing Squat Logic ---")
    state = ExerciseState("squats")
    
    # 1. Test Perfection (Top to Bottom and Back)
    # Reference median: ~79°
    # Target depth: ~99°
    print("Feeding 'perfect' squat path...")
    path = [170, 150, 120, 80, 100, 130, 170]
    for angle in path:
        # Mocking landmarks to yield specific knee angle
        # Simple triangle: hip(0,0), ankle(1,0), knee?
        # For simplicity, let's just mock the _angle helper in the instance if we could, 
        # but better to provide landmarks that calculate to these angles.
        pass

    # Actually, it's easier to mock the landmarks specifically.
    def get_lms(k_angle, sway=0):
        # hip at (0.5, 0.4), knee at (0.5, 0.6)
        # Vector BA (knee -> hip) = (0, -0.2)
        # Vector BC (knee -> ankle) must be at angle k_angle
        # We want angle(A, B, C) = k_angle
        # Using law of cosines or just rotation:
        # A = (0.5, 0.4), B = (0.5, 0.6)
        # Vector BA = (0, -0.2)
        # Vector BC = rotate BA by (180 - k_angle)
        rad = (180 - k_angle) * (math.pi / 180.0)
        # Rotation matrix: [cos -sin; sin cos]
        # v = [0, 0.2] (vector B->A flipped for 180 deg base)
        # Or just use polar coords:
        ankle_x = 0.5 + math.sin(rad) * 0.2
        ankle_y = 0.6 + math.cos(rad) * 0.2
        
        lms = {
            "left_hip": {"x": 0.5 + sway, "y": 0.4, "visibility": 0.9},
            "right_hip": {"x": 0.6 + sway, "y": 0.4, "visibility": 0.9},
            "left_knee": {"x": 0.5, "y": 0.6, "visibility": 0.9},
            "left_ankle": {"x": ankle_x, "y": ankle_y, "visibility": 0.9},
            "left_shoulder": {"x": 0.5, "y": 0.2, "visibility": 0.9}
        }
        return lms

    # Test Sway Detection
    print("Testing sway detection...")
    # Frame 1: Set baseline
    state.evaluate_frame(get_lms(170, sway=0))
    # Frame 2: Move down
    state.evaluate_frame(get_lms(140, sway=0))
    # Frame 3: Sway significantly (drift > threshold)
    # Threshold for squats is std * 2.5. std ~0.04 -> ~0.1
    res = state.evaluate_frame(get_lms(120, sway=0.15))
    print(f"Sway check: {res['mistake']} | Feedback: {res['feedback']}")

    # Test Shallow Rep
    print("Testing shallow rep detection...")
    s2 = ExerciseState("squats")
    # Threshold for squats is ~99
    # Rep: 170 -> 110 -> 170
    s2.evaluate_frame(get_lms(170))
    s2.evaluate_frame(get_lms(110)) # phase -> down
    res = s2.evaluate_frame(get_lms(170)) # phase -> up -> top
    print(f"Shallow rep check: rep_count={res['rep_count']} | Feedback: {res['feedback']} | Mistake: {res['mistake']}")

    # Test Deep Rep
    print("Testing deep rep detection...")
    s3 = ExerciseState("squats")
    s3.evaluate_frame(get_lms(170))
    s3.evaluate_frame(get_lms(80)) # below ~99
    res = s3.evaluate_frame(get_lms(170))
    print(f"Deep rep check: rep_count={res['rep_count']} | Feedback: {res['feedback']}")

if __name__ == "__main__":
    test_squat_logic()
