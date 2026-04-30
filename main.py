from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import time

# Load .env file automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

from rag_pipeline import RAGPipeline
from website.backend.cv_app import cv_router
from website.backend.sensor_handler import mqtt_handler, sensor_store
from ollama_client import OllamaClient

pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    # Initialize RAG Pipeline and Gemini
    print("[SERVER] Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    # Start MQTT Sensor Listener (Arch Linux Local Broker)
    print("[SERVER] Starting MQTT Listener...")
    mqtt_handler.start()
    yield
    # Shutdown
    print("[SERVER] Stopping MQTT Listener...")
    mqtt_handler.stop()

app = FastAPI(title="AI Form Trainer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount WebSocket router
app.include_router(cv_router)

# Initialize Ollama
llm_client = OllamaClient()

# ── Request Schemas ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_context: Optional[str] = ""

class FinalAnalysisRequest(BaseModel):
    exercise: str
    mistakes: List[dict] = [] # list of {mistake: str, timestamp: float}

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/sensor_data")
async def get_sensor_data():
    """Returns the latest frame-synced sensor data."""
    return sensor_store.get_latest()

@app.post("/chat")
async def chat_with_trainer(request: ChatRequest):
    """RAG-backed chat with the Trainer."""
    try:
        # Pull relevant context from the knowledge base
        rag_context = ""
        if pipeline:
            try:
                chunks = pipeline.retriever.retrieve_context(
                    exercise=request.session_context.split("Exercise: ")[-1].split(".")[0].strip() if "Exercise:" in (request.session_context or "") else "",
                    mistake="",
                    top_k=4
                )
                if chunks:
                    rag_context = "\n\nRelevant coaching knowledge:\n" + "\n---\n".join(chunks)
            except Exception:
                pass

        enriched_context = (request.session_context or "") + rag_context
        response = llm_client.chat(request.message, enriched_context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/final_analysis")
async def final_analysis_endpoint(request: FinalAnalysisRequest):
    """Unified session summary using CV mistakes + Sensor biometrics."""
    try:
        print("\n" + "="*60)
        print(f"[ANALYSIS] Exercise: {request.exercise}")
        print(f"[ANALYSIS] Incoming mistakes from frontend: {len(request.mistakes)}")
        for i, m in enumerate(request.mistakes):
            print(f"  [{i}] mistake={m.get('mistake','?')} feedback={m.get('feedback','?')[:60]}")

        # 1. Aggregate mistakes
        mistake_counts = {}
        for m in request.mistakes:
            key = m.get("mistake", "unknown")
            if key and key != "None":  # filter out null/None mistakes
                mistake_counts[key] = mistake_counts.get(key, 0) + 1

        mistakes_summary = [{"mistake": k, "count": v} for k, v in mistake_counts.items()]
        print(f"[ANALYSIS] Aggregated mistakes: {mistakes_summary}")

        # 2. Aggregate sensor stats
        history = sensor_store.get_history()
        waist = history.get("waist", [])

        avg_stability  = 0.0
        max_stability  = 0.0
        avg_tilt       = 0.0
        tilt_baseline  = 100.0
        momentum_events = 0

        if waist:
            stabs = [d.get("stability", 0) for d in waist]
            tilts = [d.get("tilt", 100)    for d in waist]
            if stabs:
                avg_stability   = sum(stabs) / len(stabs)
                max_stability   = max(stabs)
                momentum_events = sum(1 for s in stabs if s > 2.0)
            if tilts:
                avg_tilt       = sum(tilts) / len(tilts)
                tilt_baseline  = tilts[0]

        sensor_stats = {
            "avg_stability":        avg_stability,
            "max_stability":        max_stability,
            "avg_tilt":             avg_tilt,
            "tilt_baseline":        tilt_baseline,
            "momentum_events":      momentum_events,
            "total_sensor_readings": len(waist)
        }
        print(f"[ANALYSIS] Sensor stats: {sensor_stats}")

        # 3. Get RAG context for health/form cues
        context = []
        if pipeline:
            try:
                seen = set()
                # Pull targeted chunks for each unique mistake
                for m in mistakes_summary:
                    print(f"[RAG] Querying for exercise={request.exercise}, mistake={m['mistake']}")
                    chunks = pipeline.retriever.retrieve_context(
                        exercise=request.exercise,
                        mistake=m["mistake"],
                        top_k=3
                    )
                    print(f"[RAG]   → got {len(chunks)} chunks")
                    for c in chunks:
                        if c not in seen:
                            seen.add(c)
                            context.append(c)

                # Also pull general coaching context for the exercise
                print(f"[RAG] Querying general context for exercise={request.exercise}")
                general = pipeline.retriever.retrieve_context(
                    exercise=request.exercise,
                    mistake="",
                    top_k=4
                )
                print(f"[RAG]   → got {len(general)} general chunks")
                for c in general:
                    if c not in seen:
                        seen.add(c)
                        context.append(c)

                print(f"[RAG] Total unique context chunks: {len(context)}")
                for i, c in enumerate(context):
                    print(f"[RAG]   [{i}] {c[:100]}...")
            except Exception as rag_err:
                print(f"[RAG] Retrieval error: {rag_err}")
                import traceback; traceback.print_exc()
                context = []

        if not context:
            context = [f"General guidance for {request.exercise}: "
                       f"Maintain controlled movement, full range of motion, and proper breathing."]
            print("[RAG] WARNING: No RAG context retrieved, using fallback!")

        # 4. Generate Unified Trainer Feedback
        print(f"[OLLAMA] Sending to Ollama with {len(context)} context chunks, {len(mistakes_summary)} mistakes, sensor={sensor_stats['total_sensor_readings']} readings")
        analysis = llm_client.generate_feedback(context, request.exercise, mistakes_summary, sensor_stats)
        print(f"[OLLAMA] Response keys: {list(analysis.keys()) if isinstance(analysis, dict) else 'NOT A DICT'}")
        print(f"[OLLAMA] form_score={analysis.get('form_score','?')}, headline={str(analysis.get('headline','?'))[:80]}")
        print("="*60 + "\n")

        return analysis

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── Static File Serving ───────────────────────────────────────────────────────

if os.path.exists("videos"):
    app.mount("/videos", StaticFiles(directory="videos"), name="videos")

if os.path.exists("website/frontend"):
    app.mount("/", StaticFiles(directory="website/frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print("[SERVER] Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
