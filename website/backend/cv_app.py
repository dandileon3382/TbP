from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from website.backend.pose_extractor import PoseExtractor
from website.backend.exercise_logic import ExerciseState
import base64
import numpy as np
import cv2
import json
import traceback

cv_router = APIRouter()

@cv_router.websocket("/ws/form_detection")
async def form_detection_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    
    pose_extractor = PoseExtractor()
    state = ExerciseState(exercise="squats")
    
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            # Handle exercise selection
            if payload.get("type") == "set_exercise":
                exercise = payload.get("exercise", "squats")
                state = ExerciseState(exercise=exercise)
                print(f"[WS] Exercise set to: {exercise}")
                await websocket.send_json({"status": "exercise updated", "exercise": state.exercise})
                continue
            
            # Handle frame
            frame_data = payload.get("image")
            if not frame_data:
                continue
            
            try:
                # Decode Base64 image
                if "," in frame_data:
                    _, encoded = frame_data.split(",", 1)
                else:
                    encoded = frame_data
                    
                img_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    print("[WS] Failed to decode image")
                    await websocket.send_json({
                        "rep_count": state.rep_count,
                        "feedback": "Failed to decode frame",
                        "mistake": None,
                        "landmarks": None
                    })
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                landmarks = pose_extractor.extract_landmarks(img_rgb)
                
                if landmarks:
                    result = state.evaluate_frame(landmarks)
                    await websocket.send_json(result)
                else:
                    await websocket.send_json({
                        "rep_count": state.rep_count,
                        "feedback": "No pose detected — step back so full body is visible",
                        "mistake": None,
                        "landmarks": None
                    })
                    
            except Exception as frame_err:
                print(f"[WS] Frame processing error: {frame_err}")
                traceback.print_exc()
                await websocket.send_json({
                    "rep_count": state.rep_count,
                    "feedback": f"Processing error",
                    "mistake": None,
                    "landmarks": None
                })
                    
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Fatal error: {e}")
        traceback.print_exc()
