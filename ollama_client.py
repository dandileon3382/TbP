import os
import json
import requests

class OllamaClient:
    def __init__(self, model_name: str = "llama3.2"):
        self.base_url = "http://localhost:11434/api/generate"
        self.chat_url = "http://localhost:11434/api/chat"
        self.model_name = os.environ.get("OLLAMA_MODEL", model_name)

        # ── System instruction: structured JSON session analysis ──
        self.system_instruction = """You are Trainer — a world-class biomechanics expert and certified strength & conditioning specialist.
You are reviewing post-session data from a smart coaching system that uses both computer vision (pose detection) AND a waist-mounted IMU sensor.

Your output MUST be a valid JSON object with EXACTLY these keys:
{
  "form_score": <integer 0-100 representing overall form quality this session>,
  "headline": "<1-2 sentence honest opener capturing the session vibe>",
  "narrative": "<4-6 sentences weaving CV mistakes AND sensor data into a coherent biomechanical story. Mention specific numbers. Explain cause-and-effect chains.>",
  "sensor_analysis": "<2-3 sentences interpreting the waist IMU data specifically: what the stability score and tilt deviation reveal about core engagement, momentum usage, and compensation patterns.>",
  "per_mistake_analysis": [
    {
      "mistake": "<mistake key exactly as given>",
      "count": <number of occurrences>,
      "cause": "<biomechanical root cause — WHY does this happen? Be specific.>",
      "effect": "<performance or injury consequence if uncorrected>",
      "drill": "<ONE specific corrective drill with brief instructions (2-3 sentences)>"
    }
  ],
  "action_plan": "<3 numbered, specific coaching cues to apply in the very next session>",
  "warmup_recommendation": "<2-3 targeted warm-up movements based on the mistakes and exercise>",
  "encouragement": "<1 genuine, specific closing line acknowledging what they did well>"
}

Rules:
- Speak directly to 'you'.
- Use specific numbers from the data (degrees, counts, sensor values).
- per_mistake_analysis must have one entry per unique mistake type provided.
- If sensor data shows body_momentum or excessive_torso_swing, explain mechanically how swinging the body shifts load away from the target muscle.
- Return ONLY valid JSON. No text outside the JSON object.
"""

        # ── System instruction: conversational chat (NO JSON output) ──
        self.chat_system_instruction = """You are Trainer — a world-class biomechanics expert and personal fitness coach.
You have a warm, direct, and motivating personality. You speak naturally and conversationally, like a real coach.

When a client asks you something, give a concise, helpful answer in plain English. No bullet-point lists unless it genuinely helps clarity. No JSON. No markdown headers. Just speak like a human coach would.

Keep responses to 2-4 sentences for simple questions. Be specific about the exercise they are doing.
"""

    def generate_feedback(
        self,
        context_chunks: list[str],
        exercise: str,
        mistakes_summary: list[dict],
        sensor_stats: dict = None
    ) -> dict:
        """
        Generates a comprehensive expert coaching summary for the whole session.
        """
        context_str     = "\n\n---\n\n".join(context_chunks) if context_chunks else "No specific context available."
        readable_exercise = exercise.replace("_", " ").title()

        mistakes_text = ""
        for m in mistakes_summary:
            mistakes_text += f"  - {m['mistake'].replace('_', ' ')}: {m['count']} occurrence(s)\n"
        if not mistakes_text:
            mistakes_text = "  No mistakes logged — session appeared clean.\n"

        sensor_text = "No sensor data available (sensor not connected)."
        if sensor_stats and sensor_stats.get("total_sensor_readings", 0) > 0:
            sensor_text = (
                f"  - Average stability index: {sensor_stats.get('avg_stability', 0):.2f} "
                f"(lower is better; 0=perfectly still)\n"
                f"  - Peak stability spike: {sensor_stats.get('max_stability', 0):.2f}\n"
                f"  - Tilt baseline (vertical wear): {sensor_stats.get('tilt_baseline', 100):.1f}°\n"
                f"  - Average tilt during session: {sensor_stats.get('avg_tilt', 100):.1f}°\n"
                f"  - Momentum cheating events detected: {sensor_stats.get('momentum_events', 0)}\n"
                f"  - Total sensor readings: {sensor_stats.get('total_sensor_readings', 0)}"
            )

        prompt = f"""COACHING KNOWLEDGE BASE (use for context and drill ideas):
{context_str}

---
SESSION DATA — {readable_exercise}:

Mistakes detected by computer vision:
{mistakes_text}
Waist IMU sensor data (sensor worn vertically, tilt baseline ≌100°):
{sensor_text}

Now produce your comprehensive JSON coaching report."""

        print(f"\n[OLLAMA PROMPT]\n{prompt[:500]}...\n[END PROMPT PREVIEW]\n")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": self.system_instruction,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.65
            }
        }

        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip()

            if raw.startswith("```"):
                raw = raw.split("```", 1)[1]
                if raw.lower().startswith("json"): raw = raw[4:]
                raw = raw.rsplit("```", 1)[0].strip()

            return json.loads(raw)

        except Exception as e:
            print(f"[Ollama] generate_feedback error: {e}")
            return {
                "form_score": 65,
                "headline": "Great effort on that session!",
                "narrative": "I noticed some challenges with your form, particularly around consistency. Focus on keeping your core engaged and your movements controlled.",
                "sensor_analysis": "Sensor data was unavailable or inconclusive for this session.",
                "per_mistake_analysis": [],
                "action_plan": "1. Slow down each rep. 2. Focus on full range of motion. 3. Brace your core before every rep.",
                "warmup_recommendation": "Spend 5 minutes on dynamic stretching targeting the muscles used in this exercise.",
                "encouragement": "Every session is progress — let's refine the technique next time!"
            }

    def chat(self, user_msg: str, session_context: str) -> str:
        """Handles real-time interaction with the Trainer. Returns plain conversational text."""
        prompt_with_context = f"Session Context:\n{session_context}\n\nClient: {user_msg}\nTrainer:"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.chat_system_instruction},
                {"role": "user", "content": prompt_with_context}
            ],
            "stream": False,
            "options": {
                "temperature": 0.8
            }
        }

        try:
            response = requests.post(self.chat_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[Ollama] chat error: {e}")
            return "I'm having a bit of trouble connecting to my knowledge base right now — give me a moment and try again!"
