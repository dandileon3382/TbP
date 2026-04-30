"""
exercise_logic.py
=================
Comprehensive real-time exercise form analysis.

Every frame, multiple joint angles are evaluated against reference ranges
extracted from the tutorial videos.  Each exercise checks several
biomechanical signals, not just the primary angle, so mistakes like
hip sway, knee valgus, forward lean, elbow flare, and shallow reps are
all detected independently.

Rep counter only increments when the athlete reaches the minimum required
depth — shallow reps are rejected.
"""

from website.backend.geometry import calculate_angle
import os
import json
import time
import math

# Soft import — sensor store may not be available in isolated test runs
try:
    from website.backend.sensor_handler import sensor_store as _sensor_store
except ImportError:
    _sensor_store = None


# ── Load reference data ────────────────────────────────────────────────────────

def _load_reference(exercise: str) -> dict:
    """Return the stats dict from data/reference/<exercise>.json, or {}."""
    base = os.path.join(os.path.dirname(__file__), "..", "..", "data", "reference")
    path = os.path.abspath(os.path.join(base, f"{exercise}.json"))
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("stats", {})
    return {}


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _get_side(landmarks, joints):
    left  = sum(landmarks.get(f"left_{j}",  {}).get("visibility", 0) for j in joints)
    right = sum(landmarks.get(f"right_{j}", {}).get("visibility", 0) for j in joints)
    return "left" if left >= right else "right"

def _j(landmarks, side, name):
    return landmarks.get(f"{side}_{name}")

def _angle(a, b, c):
    """2D angle at b given dicts with x/y keys.  Returns None on bad input."""
    if not (a and b and c):
        return None
    return calculate_angle(a, b, c)

def _visible(*pts, threshold=0.3):
    return all(p and p.get("visibility", 0) > threshold for p in pts)


# ══════════════════════════════════════════════════════════════════════════════
#  ExerciseState
# ══════════════════════════════════════════════════════════════════════════════

class ExerciseState:
    def __init__(self, exercise: str = "squats"):
        self.exercise       = exercise
        self.rep_count      = 0
        self.phase          = "top"        # top → down → up → top
        self.min_angle_seen = 180.0        # deepest point reached this rep
        self.mistakes_log   = []
        self.last_mistake_time: dict[str, float] = {}

        # Baseline values set on first valid frame
        self._hip_mid_x_baseline: float | None = None
        self._frame_count = 0

        # Sensor momentum tracking
        # Sensor worn vertically on waist → tilt baseline ≈ 100°
        self._tilt_baseline: float | None = None
        self._momentum_events: int = 0

        # Load video-extracted reference
        self._ref = _load_reference(exercise)
        
        # ── Compute Dynamic Thresholds ──────────────────────────────────────────
        # We use the reference stats (median, p5, std) to define "ideal"
        # and add a buffer for human variation.
        self.thresholds = {}
        self._init_thresholds()

    def _init_thresholds(self):
        """Derive operating thresholds from video-extracted statistics."""
        if not self._ref:
            # Fallback hardcoded values if reference is missing
            self.thresholds = {
                "depth": 110.0,
                "sway": 0.06,
                "alignment": 0.25,
                "lean": 0.12,
                "body_angle": 155.0,
                "flare": 1.6
            }
            return

        # 1. Depth: Use median depth from tutorial + 15% buffer
        # For knee/elbow angles, smaller = deeper. 
        # So target = median * 1.15 (more lenient) or median * 0.85 (stricter)
        # Looking at squats.json, median was 79. 
        # Parallel is roughly 90-100. Let's use p75 as a "minimum acceptable depth"
        # because in the video p75 is 133 and median is 79.
        # Let's try (median + (max - median) * 0.3) as a balanced depth target.
        if self.exercise == "squats":
            k_stats = self._ref.get("knee_angle", {})
            self.thresholds["depth"] = k_stats.get("median", 110.0) + 20.0 # Buffer from perfect
            self.thresholds["sway"] = self._ref.get("hip_mid_x", {}).get("std", 0.04) * 2.5
            self.thresholds["alignment"] = abs(self._ref.get("knee_alignment", {}).get("p5", -9.5)) * 1.5
        elif self.exercise == "lunges":
            f_stats = self._ref.get("front_knee_angle", {})
            self.thresholds["depth"] = f_stats.get("median", 120.0) + 15.0
            self.thresholds["tilt"] = self._ref.get("hip_tilt", {}).get("p95", 0.01) * 2.0
        elif self.exercise == "pushups":
            e_stats = self._ref.get("elbow_angle", {})
            self.thresholds["depth"] = e_stats.get("median", 110.0) + 15.0
            self.thresholds["body_angle"] = self._ref.get("body_angle", {}).get("p5", 160.0) - 5.0
            self.thresholds["flare"] = self._ref.get("elbow_ratio", {}).get("median", 1.8) * 1.3
        elif self.exercise == "tricep_dips":
            e_stats = self._ref.get("elbow_angle", {})
            self.thresholds["depth"] = e_stats.get("median", 120.0) + 15.0
        elif self.exercise == "situps":
            h_stats = self._ref.get("hip_angle", {})
            self.thresholds["depth"] = h_stats.get("median", 110.0) + 15.0
            self.thresholds["neck"] = self._ref.get("neck_angle", {}).get("p95", 127.0) + 10.0

        # General fallbacks for common keys
        self.thresholds.setdefault("lean", 0.12)
        self.thresholds.setdefault("alignment", 0.25)
        self.thresholds.setdefault("sway", 0.06)

        # Sensor momentum thresholds per exercise
        # stability = (|gyrX|+|gyrY|+|gyrZ|)/100 — spikes when body swings
        # tilt_deviation = how many degrees tilt changes from standing baseline
        _momentum = {
            "squats":      {"momentum_stability": 3.0,  "tilt_deviation": 20.0},
            "lunges":      {"momentum_stability": 2.5,  "tilt_deviation": 15.0},
            "pushups":     {"momentum_stability": 2.0,  "tilt_deviation": 15.0},
            "tricep_dips": {"momentum_stability": 2.0,  "tilt_deviation": 12.0},
            "situps":      {"momentum_stability": 3.5,  "tilt_deviation": 25.0},
        }.get(self.exercise, {"momentum_stability": 2.5, "tilt_deviation": 15.0})
        self.thresholds.setdefault("momentum_stability", _momentum["momentum_stability"])
        self.thresholds.setdefault("tilt_deviation",     _momentum["tilt_deviation"])

    # ── Throttled mistake emitter ──────────────────────────────────────────────
    def _emit(self, mistake: str, feedback: str, cooldown: float = 2.5):
        now = time.time()
        if now - self.last_mistake_time.get(mistake, 0) >= cooldown:
            self.last_mistake_time[mistake] = now
            self.mistakes_log.append({
                "timestamp": round(now, 2),
                "mistake":   mistake,
                "feedback":  feedback
            })
            return mistake, feedback
        return None, None   # still cooling down — don't spam

    # ── Sensor helpers ────────────────────────────────────────────────────────
    def _get_sensor_data(self) -> dict | None:
        """Safely fetch the latest waist reading from the shared MQTT store."""
        if _sensor_store is None:
            return None
        try:
            return _sensor_store.get_latest().get("waist")
        except Exception:
            return None

    def _update_tilt_baseline(self):
        """Called each frame when phase=='top'. EMA-tracks the standing tilt."""
        if self.phase != "top":
            return
        waist = self._get_sensor_data()
        if not waist:
            return
        tilt = waist.get("tilt", 100.0)
        if self._tilt_baseline is None:
            self._tilt_baseline = tilt
        else:
            # Slow EMA — baseline drifts very gradually with the sensor warm-up
            self._tilt_baseline = 0.96 * self._tilt_baseline + 0.04 * tilt

    def _check_sensor_momentum(self) -> tuple:
        """
        Detect momentum cheating via the waist MPU6050.
        Sensor is worn vertically so tilt baseline ≈ 100°.
        Only fires during the 'down' phase of a rep.
        Returns (_emit result) or (None, None).
        """
        if self.phase != "down" or self._tilt_baseline is None:
            return None, None
        waist = self._get_sensor_data()
        if not waist:
            return None, None

        stability = waist.get("stability", 0.0)
        tilt      = waist.get("tilt",      self._tilt_baseline)
        tilt_dev  = abs(tilt - self._tilt_baseline)

        stab_limit = self.thresholds.get("momentum_stability", 2.5)
        tilt_limit = self.thresholds.get("tilt_deviation",     15.0)

        if stability > stab_limit:
            self._momentum_events += 1
            return self._emit(
                "body_momentum",
                f"Stop swinging! You're using body momentum instead of muscle strength "
                f"(sensor gyro: {stability:.1f}×). Slow down and squeeze through the movement.",
                cooldown=3.0
            )
        if tilt_dev > tilt_limit:
            self._momentum_events += 1
            return self._emit(
                "excessive_torso_swing",
                f"Your waist tilted {tilt_dev:.0f}° off your start position — "
                f"brace your core and stop swinging your torso.",
                cooldown=3.0
            )
        return None, None

    # ══════════════════════════════════════════════════════════════════════════
    #  Main evaluation dispatcher
    # ══════════════════════════════════════════════════════════════════════════
    def evaluate_frame(self, landmarks: dict) -> dict:
        self._frame_count += 1
        feedback = "Good form ✓"
        mistake  = None

        # Update sensor tilt baseline while standing between reps
        self._update_tilt_baseline()

        if self.exercise == "squats":
            feedback, mistake = self._eval_squats(landmarks)
        elif self.exercise == "lunges":
            feedback, mistake = self._eval_lunges(landmarks)
        elif self.exercise == "pushups":
            feedback, mistake = self._eval_pushups(landmarks)
        elif self.exercise == "tricep_dips":
            feedback, mistake = self._eval_tricep_dips(landmarks)
        elif self.exercise == "situps":
            feedback, mistake = self._eval_situps(landmarks)

        return {
            "rep_count":       self.rep_count,
            "feedback":        feedback,
            "mistake":         mistake,
            "landmarks":       landmarks,
            "momentum_events": self._momentum_events
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  SQUATS
    # ══════════════════════════════════════════════════════════════════════════
    def _eval_squats(self, lms):
        side = _get_side(lms, ["hip", "knee", "ankle"])
        hip   = _j(lms, side, "hip")
        knee  = _j(lms, side, "knee")
        ankle = _j(lms, side, "ankle")
        sh    = _j(lms, side, "shoulder")

        if not _visible(hip, knee, ankle):
            return "Step back so your full body is visible", None

        knee_angle = _angle(hip, knee, ankle)
        feedback = f"Knee: {knee_angle:.0f}°"

        # ── Rep phase machine ────────────────────────────────────────────────
        if self.phase == "top" and knee_angle < 155:
            self.phase = "down"
            self.min_angle_seen = knee_angle

        if self.phase == "down":
            self.min_angle_seen = min(self.min_angle_seen, knee_angle)
            if knee_angle > self.min_angle_seen + 20:
                # Starting to come back up
                self.phase = "up"

        if self.phase == "up" and knee_angle > 160:
            target_depth = self.thresholds.get("depth", 110.0)
            if self.min_angle_seen < target_depth:
                self.rep_count += 1
                feedback = f"Rep {self.rep_count} complete! ✓"
            else:
                feedback = f"Shallow rep — go lower! (reached {self.min_angle_seen:.0f}°)"
                m, fb = self._emit("shallow_squat", f"Squat deeper — try to reach {target_depth:.0f}° at the bottom")
                if m:
                    return fb, m
            self.phase = "top"
            self.min_angle_seen = 180.0
            return feedback, None

        # ── Multi-mistake detection every frame ──────────────────────────────

        # 1. Hip sway left/right
        lh = lms.get("left_hip"); rh = lms.get("right_hip")
        if lh and rh:
            mid_x = (lh["x"] + rh["x"]) / 2
            if self._hip_mid_x_baseline is None and self.phase == "top":
                self._hip_mid_x_baseline = mid_x
            if self._hip_mid_x_baseline is not None:
                drift = abs(mid_x - self._hip_mid_x_baseline)
                sway_limit = self.thresholds.get("sway", 0.06)
                if drift > sway_limit and self.phase == "down":
                    m, fb = self._emit("hip_sway", f"Your hips are swaying sideways ({drift*100:.1f}% drift) — keep them centred")
                    if m: return fb, m

        # 2. Knees caving in (valgus)
        opp_side = "right" if side == "left" else "left"
        opp_knee  = _j(lms, opp_side, "knee")
        opp_ankle = _j(lms, opp_side, "ankle")
        if knee and ankle and opp_knee and opp_ankle and self.phase == "down":
            hip_ankle_w = abs(hip["x"] - ankle["x"]) + 1e-6
            lateral = (knee["x"] - ankle["x"]) / hip_ankle_w
            align_limit = self.thresholds.get("alignment", 0.25)
            if (side == "left" and lateral < -align_limit) or (side == "right" and lateral > align_limit):
                m, fb = self._emit("knees_caving_in", "Knees caving in — push them out in line with your toes")
                if m: return fb, m

        # 3. Forward lean
        if sh and hip and self.phase == "down":
            lean = abs(sh["x"] - hip["x"])
            lean_limit = self.thresholds.get("lean", 0.12)
            if lean > lean_limit:
                m, fb = self._emit("forward_lean", "Keep your chest up — you're leaning too far forward")
                if m: return fb, m

        # 4. Too deep
        if self.phase == "down" and knee_angle < 50:
            m, fb = self._emit("too_deep", f"Too deep! ({knee_angle:.0f}°) — stop just at or below parallel")
            if m: return fb, m

        # Sensor: body momentum check
        m, fb = self._check_sensor_momentum()
        if m: return fb, m

        return feedback, None


    # ══════════════════════════════════════════════════════════════════════════
    #  LUNGES
    # ══════════════════════════════════════════════════════════════════════════
    def _eval_lunges(self, lms):
        la = lms.get("left_ankle",  {"x": 0.5, "y": 0.5, "visibility": 0})
        ra = lms.get("right_ankle", {"x": 0.5, "y": 0.5, "visibility": 0})

        # Front foot is the one lower on screen (higher y in image coords)
        fside = "left" if la.get("y", 0) > ra.get("y", 0) else "right"
        bside = "right" if fside == "left" else "left"

        fhip   = _j(lms, fside, "hip")
        fknee  = _j(lms, fside, "knee")
        fankle = _j(lms, fside, "ankle")
        fsh    = _j(lms, fside, "shoulder")

        if not _visible(fhip, fknee, fankle):
            return "Step back so your full body is visible", None

        front_angle = _angle(fhip, fknee, fankle)
        feedback = f"Front knee: {front_angle:.0f}°"

        # Rep phase
        if self.phase == "top" and front_angle < 160:
            self.phase = "down"
            self.min_angle_seen = front_angle

        if self.phase == "down":
            self.min_angle_seen = min(self.min_angle_seen, front_angle)
            if front_angle > self.min_angle_seen + 20:
                self.phase = "up"

        if self.phase == "up" and front_angle > 160:
            target_depth = self.thresholds.get("depth", 120.0)
            if self.min_angle_seen < target_depth:
                self.rep_count += 1
                feedback = f"Rep {self.rep_count} complete! ✓"
            else:
                feedback = f"Shallow lunge — go lower! (reached {self.min_angle_seen:.0f}°)"
                m, fb = self._emit("shallow_lunge", f"Drop your back knee lower — aim for {target_depth:.0f}° front knee angle")
                if m: return fb, m
            self.phase = "top"
            self.min_angle_seen = 180.0
            return feedback, None

        # Knee collapse
        if fknee and fankle and fhip and self.phase == "down":
            hip_ankle_w = abs(fhip["x"] - fankle["x"]) + 1e-6
            lateral = (fknee["x"] - fankle["x"]) / hip_ankle_w
            align_limit = self.thresholds.get("alignment", 0.25)
            if (fside == "left" and lateral < -align_limit) or (fside == "right" and lateral > align_limit):
                m, fb = self._emit("knee_collapse", "Front knee caving in — push it out over your toes")
                if m: return fb, m

        # Forward lean
        if fsh and fhip and self.phase == "down":
            lean_limit = self.thresholds.get("lean", 0.12)
            if abs(fsh["x"] - fhip["x"]) > lean_limit:
                m, fb = self._emit("leaning_forward", "Keep your torso upright — you're leaning too far forward")
                if m: return fb, m

        # Hip not level
        lh = lms.get("left_hip"); rh = lms.get("right_hip")
        if lh and rh:
            tilt = abs(lh["y"] - rh["y"])
            tilt_limit = self.thresholds.get("tilt", 0.06)
            if tilt > tilt_limit and self.phase == "down":
                m, fb = self._emit("uneven_step", f"Keep your hips level — one hip is dropping ({tilt*100:.1f}%)")
                if m: return fb, m

        # Sensor: body momentum check
        m, fb = self._check_sensor_momentum()
        if m: return fb, m

        return feedback, None

    # ══════════════════════════════════════════════════════════════════════════
    #  PUSHUPS
    # ══════════════════════════════════════════════════════════════════════════
    def _eval_pushups(self, lms):
        side = _get_side(lms, ["shoulder", "elbow", "wrist"])
        sh    = _j(lms, side, "shoulder")
        el    = _j(lms, side, "elbow")
        wr    = _j(lms, side, "wrist")
        hip   = _j(lms, side, "hip")
        ankle = _j(lms, side, "ankle")

        if not _visible(sh, el, wr):
            return "Make sure your upper body is visible", None

        elbow_angle = _angle(sh, el, wr)
        feedback = f"Elbow: {elbow_angle:.0f}°"

        # Rep phase
        if self.phase == "top" and elbow_angle < 155:
            self.phase = "down"
            self.min_angle_seen = elbow_angle

        if self.phase == "down":
            self.min_angle_seen = min(self.min_angle_seen, elbow_angle)
            if elbow_angle > self.min_angle_seen + 20:
                self.phase = "up"

        if self.phase == "up" and elbow_angle > 155:
            target_depth = self.thresholds.get("depth", 110.0)
            if self.min_angle_seen < target_depth:
                self.rep_count += 1
                feedback = f"Rep {self.rep_count} complete! ✓"
            else:
                feedback = f"Too shallow — chest must near the floor! (reached {self.min_angle_seen:.0f}°)"
                m, fb = self._emit("shallow_depth", f"Lower your chest more — reach for {target_depth:.0f}°")
                if m: return fb, m
            self.phase = "top"
            self.min_angle_seen = 180.0
            return feedback, None

        # Hip sagging / high hips
        if _visible(sh, hip, ankle, threshold=0.25):
            body_angle = _angle(sh, hip, ankle)
            limit = self.thresholds.get("body_angle", 155.0)
            if body_angle is not None and body_angle < limit and self.phase == "down":
                m, fb = self._emit("hip_sagging", "Hips are sagging — squeeze your glutes to keep a straight body")
                if m: return fb, m
            if body_angle is not None and body_angle < (limit - 15) and self.phase == "top":
                m, fb = self._emit("hips_high", "Hips too high — lower them for a straight line")
                if m: return fb, m

        # Elbow flare
        ls = lms.get("left_shoulder"); rs = lms.get("right_shoulder")
        le = lms.get("left_elbow");    re = lms.get("right_elbow")
        if ls and rs and le and re:
            sh_w = abs(ls["x"] - rs["x"])
            el_w = abs(le["x"] - re["x"])
            flare_limit = self.thresholds.get("flare", 1.6)
            if sh_w > 0.05 and el_w / sh_w > flare_limit and self.phase == "down":
                m, fb = self._emit("elbow_flare", "Elbows flaring wide — tuck them in slightly")
                if m: return fb, m

        # Sensor: body momentum check
        m, fb = self._check_sensor_momentum()
        if m: return fb, m

        return feedback, None

    # ══════════════════════════════════════════════════════════════════════════
    #  TRICEP DIPS
    # ══════════════════════════════════════════════════════════════════════════
    def _eval_tricep_dips(self, lms):
        side = _get_side(lms, ["shoulder", "elbow", "wrist"])
        sh = _j(lms, side, "shoulder")
        el = _j(lms, side, "elbow")
        wr = _j(lms, side, "wrist")
        hip = _j(lms, side, "hip")

        if not _visible(sh, el, wr):
            return "Make sure your upper body is visible", None

        elbow_angle = _angle(sh, el, wr)
        feedback = f"Elbow: {elbow_angle:.0f}°"

        # Rep phase
        if self.phase == "top" and elbow_angle < 155:
            self.phase = "down"
            self.min_angle_seen = elbow_angle

        if self.phase == "down":
            self.min_angle_seen = min(self.min_angle_seen, elbow_angle)
            if elbow_angle > self.min_angle_seen + 20:
                self.phase = "up"

        if self.phase == "up" and elbow_angle > 155:
            target_depth = self.thresholds.get("depth", 120.0)
            if self.min_angle_seen < target_depth:
                self.rep_count += 1
                feedback = f"Rep {self.rep_count} complete! ✓"
            else:
                feedback = f"Not deep enough — aim for 90° elbow! (reached {self.min_angle_seen:.0f}°)"
                m, fb = self._emit("shallow_depth", f"Lower yourself until your elbows reach {target_depth:.0f}°")
                if m: return fb, m
            self.phase = "top"
            self.min_angle_seen = 180.0
            return feedback, None

        # Too deep
        if self.phase == "down" and elbow_angle < 60:
            m, fb = self._emit("too_deep", "Too deep! — stop at 90° to protect your shoulders")
            if m: return fb, m

        # Elbow flare
        ls = lms.get("left_shoulder"); rs = lms.get("right_shoulder")
        le = lms.get("left_elbow");    re = lms.get("right_elbow")
        if ls and rs and le and re:
            sh_w = abs(ls["x"] - rs["x"])
            el_w = abs(le["x"] - re["x"])
            if sh_w > 0.04 and el_w / sh_w > 1.5 and self.phase == "down":
                m, fb = self._emit("elbow_flare", "Elbows flaring out — keep them pointing straight back")
                if m: return fb, m

        if sh and hip:
            if sh["y"] < 0.35 and self.phase == "down": 
                m, fb = self._emit("shoulder_shrugging", "Keep your shoulders down — don't shrug")
                if m: return fb, m

        # Sensor: body momentum check
        m, fb = self._check_sensor_momentum()
        if m: return fb, m

        return feedback, None

    # ══════════════════════════════════════════════════════════════════════════
    #  SIT-UPS
    # ══════════════════════════════════════════════════════════════════════════
    def _eval_situps(self, lms):
        side = _get_side(lms, ["shoulder", "hip", "knee"])
        sh   = _j(lms, side, "shoulder")
        hip  = _j(lms, side, "hip")
        knee = _j(lms, side, "knee")
        ear  = _j(lms, side, "ear")

        if not _visible(sh, hip, knee):
            return "Full body must be visible while lying down", None

        hip_angle = _angle(sh, hip, knee)
        feedback = f"Hip angle: {hip_angle:.0f}°"

        if self.phase == "top" and hip_angle < 150:
            self.phase = "down"
            self.min_angle_seen = hip_angle

        if self.phase == "down":
            self.min_angle_seen = min(self.min_angle_seen, hip_angle)
            if hip_angle > self.min_angle_seen + 20:
                self.phase = "up"

        if self.phase == "up" and hip_angle > 155:
            target_depth = self.thresholds.get("depth", 130.0)
            if self.min_angle_seen < target_depth:
                self.rep_count += 1
                feedback = f"Rep {self.rep_count} complete! ✓"
            else:
                feedback = f"Too shallow — crunch all the way up! (reached {self.min_angle_seen:.0f}°)"
                m, fb = self._emit("shallow_range", f"Crunch further — aim for {target_depth:.0f}° torso angle")
                if m: return fb, m
            self.phase = "top"
            self.min_angle_seen = 180.0
            return feedback, None

        # Neck pulling
        if ear and sh and self.phase == "down":
            neck_limit = self.thresholds.get("neck", 137.0)
            neck_angle = _angle(ear, sh, hip)
            if neck_angle and neck_angle < neck_limit:
                m, fb = self._emit("neck_pulling", "Keep your neck relaxed — don't pull your head forward")
                if m: return fb, m

        # Asymmetry
        ls = lms.get("left_shoulder"); rs = lms.get("right_shoulder")
        if ls and rs and self.phase == "down":
            tilt = abs(ls["y"] - rs["y"])
            if tilt > 0.07:
                m, fb = self._emit("asymmetry", "You're twisting — keep both shoulders rising equally")
                if m: return fb, m

        # Sensor: body momentum check
        m, fb = self._check_sensor_momentum()
        if m: return fb, m

        return feedback, None

