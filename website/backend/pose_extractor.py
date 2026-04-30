import mediapipe as mp
import numpy as np
import os

class PoseExtractor:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        # Find model file
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'pose_landmarker_lite.task')
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pose model not found at {model_path}")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        # Build name map from PoseLandmark enum
        self.landmark_names = {lm.value: lm.name.lower() for lm in mp.tasks.vision.PoseLandmark}

    def extract_landmarks(self, image_rgb):
        """
        Process an RGB numpy image and return pose landmarks as a dict,
        or None if no pose detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        pose = result.pose_landmarks[0]  # First person
        landmarks = {}

        for idx, lm in enumerate(pose):
            name = self.landmark_names.get(idx, f"point_{idx}")
            landmarks[name] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility if hasattr(lm, 'visibility') else lm.presence
            }

        return landmarks
