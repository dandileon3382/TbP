"""
extract_reference.py
====================
Processes each tutorial video through MediaPipe Pose to extract ideal joint-angle
statistics that represent "perfect form".  Outputs per-exercise JSON reference
files to data/reference/.

Run from the project root:
    python scripts/extract_reference.py
"""

import os
import sys
import json
import math
import numpy as np
import cv2

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mediapipe as mp

# ── MediaPipe setup ────────────────────────────────────────────────────────────
BaseOptions         = mp.tasks.BaseOptions
PoseLandmarker      = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode         = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task")
)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)

LANDMARK_NAMES = {lm.value: lm.name.lower() for lm in mp.tasks.vision.PoseLandmark}


# ── Geometry helpers ───────────────────────────────────────────────────────────

def lm(landmarks, name):
    return landmarks.get(name)

def angle(a, b, c):
    """2D angle at point b."""
    if not (a and b and c):
        return None
    ax, ay = a["x"] - b["x"], a["y"] - b["y"]
    cx, cy = c["x"] - b["x"], c["y"] - b["y"]
    dot  = ax*cx + ay*cy
    mag  = math.hypot(ax, ay) * math.hypot(cx, cy)
    if mag < 1e-8:
        return None
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))

def choose_side(landmarks, joints):
    left  = sum(landmarks.get(f"left_{j}",  {}).get("visibility", 0) for j in joints)
    right = sum(landmarks.get(f"right_{j}", {}).get("visibility", 0) for j in joints)
    return "left" if left >= right else "right"


# ── Per-exercise angle extractors ─────────────────────────────────────────────

def extract_squats(lms):
    side = choose_side(lms, ["hip", "knee", "ankle"])
    knee_a  = angle(lms.get(f"{side}_hip"),  lms.get(f"{side}_knee"),  lms.get(f"{side}_ankle"))
    back_a  = angle(lms.get(f"{side}_shoulder"), lms.get(f"{side}_hip"), lms.get(f"{side}_knee"))

    # Hip sway: midpoint x drift encoded as |left_hip.x - right_hip.x| change
    lh = lms.get("left_hip"); rh = lms.get("right_hip")
    hip_mid_x = ((lh["x"] + rh["x"]) / 2) if lh and rh else None

    # Knee alignment: ideal knee.x should track ankle.x closely
    knee = lms.get(f"{side}_knee"); ankle = lms.get(f"{side}_ankle")
    hip  = lms.get(f"{side}_hip")
    knee_in = None
    if knee and ankle and hip:
        # normalised lateral deviation: positive = knee inside ankle (valgus)
        direction = hip["x"] - ankle["x"]  # sign indicates which side
        knee_in   = (knee["x"] - ankle["x"]) / (abs(direction) + 1e-6)

    return {
        "knee_angle": knee_a,
        "back_angle": back_a,
        "hip_mid_x": hip_mid_x,
        "knee_alignment": knee_in,
    }

def extract_lunges(lms):
    # Front leg = ankle with lower y (higher on screen = further forward in lunge)
    la = lms.get("left_ankle", {"y": 0.5}); ra = lms.get("right_ankle", {"y": 0.5})
    front = "left" if la.get("y", 0) > ra.get("y", 0) else "right"
    back  = "right" if front == "left" else "left"

    fknee_a = angle(lms.get(f"{front}_hip"), lms.get(f"{front}_knee"), lms.get(f"{front}_ankle"))
    back_a  = angle(lms.get(f"{front}_shoulder"), lms.get(f"{front}_hip"), lms.get(f"{front}_knee"))

    lh = lms.get("left_hip"); rh = lms.get("right_hip")
    hip_tilt = abs(lh["y"] - rh["y"]) if lh and rh else None

    return {
        "front_knee_angle": fknee_a,
        "back_angle": back_a,
        "hip_tilt": hip_tilt,
    }

def extract_pushups(lms):
    side = choose_side(lms, ["shoulder", "elbow", "wrist"])
    elbow_a = angle(lms.get(f"{side}_shoulder"), lms.get(f"{side}_elbow"), lms.get(f"{side}_wrist"))
    body_a  = angle(lms.get(f"{side}_shoulder"), lms.get(f"{side}_hip"), lms.get(f"{side}_ankle"))

    # Elbow flare ratio: elbow width vs shoulder width
    ls = lms.get("left_shoulder"); rs = lms.get("right_shoulder")
    le = lms.get("left_elbow");    re = lms.get("right_elbow")
    elbow_ratio = None
    if ls and rs and le and re:
        sh_w = abs(ls["x"] - rs["x"])
        el_w = abs(le["x"] - re["x"])
        elbow_ratio = el_w / (sh_w + 1e-6)

    return {
        "elbow_angle": elbow_a,
        "body_angle": body_a,
        "elbow_ratio": elbow_ratio,
    }

def extract_tricep_dips(lms):
    side = choose_side(lms, ["shoulder", "elbow", "wrist"])
    elbow_a    = angle(lms.get(f"{side}_shoulder"), lms.get(f"{side}_elbow"), lms.get(f"{side}_wrist"))
    sh_elbow_a = angle(lms.get(f"{side}_hip"), lms.get(f"{side}_shoulder"), lms.get(f"{side}_elbow"))
    return {
        "elbow_angle": elbow_a,
        "shoulder_angle": sh_elbow_a,
    }

def extract_situps(lms):
    side = choose_side(lms, ["shoulder", "hip", "knee"])
    hip_a = angle(lms.get(f"{side}_shoulder"), lms.get(f"{side}_hip"), lms.get(f"{side}_knee"))

    # Neck: ear relative to shoulder — proxy for neck pull
    ear = lms.get(f"{side}_ear"); sh = lms.get(f"{side}_shoulder"); hip = lms.get(f"{side}_hip")
    neck_a = angle(ear, sh, hip) if ear and sh and hip else None

    return {
        "hip_angle": hip_a,
        "neck_angle": neck_a,
    }

EXTRACTORS = {
    "squats":      extract_squats,
    "lunges":      extract_lunges,
    "pushups":     extract_pushups,
    "tricep_dips": extract_tricep_dips,
    "situps":      extract_situps,
}

VIDEO_MAP = {
    "squats":      "squats.mp4",
    "lunges":      "lunges.mp4",
    "pushups":     "pushups.mp4",
    "tricep_dips": "dips.mp4",
    "situps":      "situps.mp4",
}


# ── Frame processor ────────────────────────────────────────────────────────────

def process_video(video_path: str, exercise: str) -> dict:
    extractor_fn = EXTRACTORS[exercise]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ Cannot open {video_path}")
        return {}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"  Frames: {total}  FPS: {fps:.1f}")

    # Collect all angle values per key
    records: dict[str, list[float]] = {}
    frame_idx = 0
    sample_every = max(1, int(fps / 6))  # ~6 frames per second

    with PoseLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % sample_every != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                continue

            pose = result.pose_landmarks[0]
            landmarks = {}
            for idx, lm_pt in enumerate(pose):
                name = LANDMARK_NAMES.get(idx, f"point_{idx}")
                landmarks[name] = {
                    "x": lm_pt.x, "y": lm_pt.y, "z": lm_pt.z,
                    "visibility": lm_pt.visibility if hasattr(lm_pt, "visibility") else lm_pt.presence
                }

            angles = extractor_fn(landmarks)
            for key, val in angles.items():
                if val is not None and not math.isnan(val):
                    records.setdefault(key, []).append(val)

    cap.release()

    if not records:
        print("  ✗ No pose data extracted")
        return {}

    # ── Compute statistics ─────────────────────────────────────────────────────
    stats = {}
    for key, vals in records.items():
        arr = np.array(vals)
        p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
        stats[key] = {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr)),
            "p5":     float(p5),
            "p25":    float(p25),
            "median": float(p50),
            "p75":    float(p75),
            "p95":    float(p95),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "n":      len(vals),
        }
        print(f"    {key}: median={p50:.1f}  range=[{arr.min():.1f}, {arr.max():.1f}]  n={len(vals)}")

    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    root     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vid_dir  = os.path.join(root, "videos")
    ref_dir  = os.path.join(root, "data", "reference")
    os.makedirs(ref_dir, exist_ok=True)

    for exercise, filename in VIDEO_MAP.items():
        video_path = os.path.join(vid_dir, filename)
        if not os.path.exists(video_path):
            print(f"\n[SKIP] {exercise} — video not found: {video_path}")
            continue

        print(f"\n[{exercise.upper()}] Processing {filename} ...")
        stats = process_video(video_path, exercise)

        if stats:
            out_path = os.path.join(ref_dir, f"{exercise}.json")
            with open(out_path, "w") as f:
                json.dump({"exercise": exercise, "stats": stats}, f, indent=2)
            print(f"  ✓ Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
