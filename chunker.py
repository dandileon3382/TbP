from typing import List, Dict, Any


def chunk_exercise_data(exercise_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts a single exercise JSON into rich, semantically dense text chunks.

    Chunk types produced:
      - description     : what the exercise is and why form matters
      - cues            : plain-language coaching cues (all in one chunk)
      - rules           : movement rules
      - constraints     : things to avoid
      - mistake_explanation : one detailed chunk per mistake (most valuable for RAG)
      - joint_range     : one chunk per ideal joint angle range
      - sensor_rules    : MPU6050 sensor expectations (kept for future use)
    """
    chunks = []
    ex = exercise_data.get("exercise", "unknown")

    # ── 1. Description ─────────────────────────────────────────────────────────
    desc = exercise_data.get("description", "")
    if desc:
        chunks.append({
            "text": f"Exercise: {ex}\nDescription: {desc}",
            "type": "description",
            "mistake": ""
        })

    # ── 2. Coaching Cues ───────────────────────────────────────────────────────
    cues = exercise_data.get("cues", [])
    if cues:
        cues_text = (
            f"Exercise: {ex}\n"
            f"Coaching cues for proper form:\n"
            + "\n".join(f"  • {c}" for c in cues)
        )
        chunks.append({
            "text": cues_text,
            "type": "cues",
            "mistake": ""
        })

    # ── 3. Rules ───────────────────────────────────────────────────────────────
    rules = exercise_data.get("rules", [])
    if rules:
        rules_text = (
            f"Exercise: {ex}\n"
            f"Form rules:\n"
            + "\n".join(f"  • {r}" for r in rules)
        )
        chunks.append({
            "text": rules_text,
            "type": "rules",
            "mistake": ""
        })

    # ── 4. Constraints ─────────────────────────────────────────────────────────
    constraints = exercise_data.get("constraints", [])
    if constraints:
        constraints_text = (
            f"Exercise: {ex}\n"
            f"What to avoid (constraints):\n"
            + "\n".join(f"  • {c}" for c in constraints)
        )
        chunks.append({
            "text": constraints_text,
            "type": "constraints",
            "mistake": ""
        })

    # ── 5. Per-Mistake Explanations (highest RAG value) ────────────────────────
    mistake_explanations = exercise_data.get("mistake_explanations", {})
    for mistake_key, explanation in mistake_explanations.items():
        cause = explanation.get("cause", "")
        consequence = explanation.get("consequence", "")
        fix = explanation.get("fix", "")

        mistake_text = (
            f"Exercise: {ex}\n"
            f"Mistake: {mistake_key.replace('_', ' ')}\n"
            f"Why it happens: {cause}\n"
            f"Why it is harmful: {consequence}\n"
            f"How to fix it: {fix}"
        )
        chunks.append({
            "text": mistake_text,
            "type": "mistake_explanation",
            "mistake": mistake_key
        })

    # ── 6. Joint Angle Ranges ──────────────────────────────────────────────────
    ideal_ranges = exercise_data.get("ideal_ranges", {})
    for joint, metrics in ideal_ranges.items():
        parts = []
        if "min" in metrics and "max" in metrics:
            parts.append(f"range {metrics['min']}–{metrics['max']}°")
        if "optimal" in metrics:
            parts.append(f"optimal {metrics['optimal']}°")
        if "buffer" in metrics:
            parts.append(f"acceptable buffer ±{metrics['buffer']}°")

        joint_text = (
            f"Exercise: {ex}\n"
            f"Joint: {joint.replace('_', ' ')}\n"
            f"Ideal angle: {', '.join(parts)}"
        )
        chunks.append({
            "text": joint_text,
            "type": "joint_range",
            "mistake": ""
        })

    # ── 7. Sensor Rules (stored for future hardware use) ───────────────────────
    sensor_rules = exercise_data.get("sensor_rules", {})
    if sensor_rules:
        sensor_lines = []
        for sensor_name, s in sensor_rules.items():
            note = s.get("note", "")
            unit = s.get("unit", "")
            optimal = s.get("optimal", "")
            sensor_lines.append(
                f"  • {sensor_name} ({unit}): optimal={optimal}. {note}"
            )
        sensor_text = (
            f"Exercise: {ex}\n"
            f"MPU6050 sensor ideal readings:\n"
            + "\n".join(sensor_lines)
        )
        chunks.append({
            "text": sensor_text,
            "type": "sensor_rules",
            "mistake": ""
        })

    return chunks


def process_all_exercises(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes all exercise dicts into a flat list of enriched chunk dicts.
    Each chunk has: text, type, exercise, mistake.
    """
    all_chunks = []
    for exercise_data in data:
        ex_name = exercise_data.get("exercise", "unknown")
        chunks = chunk_exercise_data(exercise_data)
        for chunk in chunks:
            chunk["exercise"] = ex_name
            all_chunks.append(chunk)

    print(f"Total chunks generated: {len(all_chunks)}")
    return all_chunks
