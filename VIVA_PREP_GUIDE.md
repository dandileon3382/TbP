# AI Form Coach: Viva Preparation Guide

This document is structured to help you easily recall and explain every technical block, architectural decision, and unique feature of the AI Form Coach project during your Viva.

---

## 1. System Architecture overview

The project uses a **Moduler, Sensor-Fusion Architecture** combined with **Local Edge AI**. 

### **Frontend (Client)**
* **Tech:** HTML, CSS, Vanilla JavaScript.
* **Functionality:** Accesses the webcam and runs Google's **MediaPipe Pose Detection** entirely in the browser. It extracts (x, y, z) coordinates for 33 body landmarks in real-time. It communicates real-time mistakes via WebSockets to the backend.

### **Backend (Server & Rules Engine)**
* **Tech:** Python, FastAPI, WebSockets.
* **Functionality:** Handles live WebSocket streams. Runs the `exercise_logic.py` engine which calculates joint angles (e.g., knee hinge, hip hinge) using 2D geometry (cosine rule / dot product) to count reps, segment phases (eccentric/concentric), and detect visual form mistakes (e.g., knee collapse, leaning forward).

### **Hardware / IoT (Sensor Fusion)**
* **Tech:** ESP32 Microcontroller + MPU6050 (Accelerometer & Gyroscope) + MQTT Protocol (Mosquitto).
* **Functionality:** Worn on the waist. The ESP32 streams 6-axis IMU data over a local WiFi MQTT broker to the Python backend. It calculates a "Stability Score" and tracks exactly how much the torso tilts.

### **LLM & RAG Pipeline (The "Trainer")**
* **Tech:** ChromaDB (Vector DB), Sentence-Transformers (Local Embeddings), Ollama + Llama 3.2 (Local LLM).
* **Functionality:** Instead of giving generic advice, the system uses **Retrieval-Augmented Generation (RAG)** to pull expert biomechanical corrections specific to the mistakes made by the user, weaving CV and IMU sensor data into an easy-to-understand coaching summary.

---

## 2. Why use the MPU6050? (The "Sensor Fusion" Argument)

**Viva Question:** *"Why do you need an external hardware sensor if you already have a camera and MediaPipe?"*

**Answer:** 
Computer Vision (CV) is excellent for tracking 2D joint angles and overall limb positions, but it has severe limitations:
1. **Depth Perception & Occlusion:** A single camera struggles to accurately measure 3D depth or track limbs that cross over each other.
2. **Micro-Movements & Core Engagement:** CV cannot easily measure "momentum" or internal core stability.
3. **The MPU6050 solves this:** Strapped to the waist, the MPU6050 acts as the body's center of gravity monitor. It detects exactly when a user "cheats" by using a sudden burst of momentum (acceleration spikes) or leans off-axis (gyroscope data). 
**Conclusion:** By fusing CV (Visual geometry) with IMU (Physical momentum/stability), the application achieves coaching accuracy close to what physical smart-wearables provide, avoiding the blind spots of purely camera-based apps.

---

## 3. How the RAG (Retrieval-Augmented Generation) Pipeline Works

**Viva Question:** *"Explain how your AI generates the feedback and what data you use."*

**Answer: The 4-Step RAG Process:**
1. **Data Ingestion:** We constructed a custom Knowledge Base (stored as JSON files inside the `data/` folder). This data acts as our "Expert Coach Brain", containing exact definitions, biomechanical causes, and corrective drills for various exercise mistakes.
2. **Embedding:** On startup, `sentence-transformers` (`all-MiniLM-L6-v2`) converts this text data into high-dimensional vector embeddings and stores them in **ChromaDB**.
3. **Retrieval (`retriever.py`):** When the user finishes a session, the backend gathers the list of mistakes (e.g., `knee_collapse`). The retriever searches ChromaDB and fetches specifically the exact paragraphs explaining *why* knee collapse happens and *how* to fix it.
4. **Generation (`ollama_client.py`):** This retrieved knowledge, along with the raw numbers from the MPU6050 (e.g., "Stability index: 2.4") and the CV app, is injected into a prompt and sent to **Ollama (Llama 3.2)**. The LLM simply formats and synthesizes this hard data into a human-readable, highly accurate JSON response without hallucinating.

---

## 4. Unique Selling Points (USPs)

If asked: *"What makes your project different from other fitness apps on the market?"*

1. **True Sensor-Fusion:** We don't rely solely on a camera, nor solely on a smartwatch. Fusing CV with an IoT waist sensor provides a complete biomechanical picture.
2. **100% Privacy & Local Execution:** Everything—from the MediaPipe vision tracking to the RAG database, to the GenAI (Ollama)—runs locally on the machine. No video feeds or health data ever touch the cloud.
3. **Anti-Hallucination AI:** Because of the strict RAG implementation, the LLM is forced to quote our structured JSON knowledge base. It won't invent fake exercises or unsafe advice.
4. **Dynamic Context-Aware Chat:** Users can chat with the virtual trainer after a session. The chat isn't a blank slate; it actively "knows" exactly what mistakes the user just made because of the injected context.

---

## 5. Future Scope / What we can add next

If asked: *"How would you improve this further?"*

1. **Full-Body IoT Network:** Expand from a single waist sensor to wearable wrist and ankle bands (BLE) to cross-validate arm and leg trajectories with the camera.
2. **Long-Term Memory (User Profiles):** Integrate the RAG system with a SQL database to remember a specific user's historical mistakes, adjusting its strictness over time.
3. **Smartwatch Integration:** Instead of a custom ESP32, pull accelerometer data directly from Apple Watch APIs or WearOS.
4. **Gamification Engine:** Develop a leaderboard system based on the generated "Form Score."

---

## 6. Quick-Fire Q&A Cheat Sheet

* **What protocol connects the sensor?** MQTT (Mosquitto Broker) over local WiFi, chosen because it is lightweight and handles real-time IoT streams flawlessly without blocking the main server.
* **Why FastAPI?** Because it natively supports `async`/`await` and WebSockets, which are absolutely critical for streaming live 30FPS video coordinate data without lag.
* **Why did you switch to Ollama?** Moving from Cloud APIs to Ollama removes the dependency on cloud API keys, eliminates rate-limiting, keeps biometrics entirely private, and guarantees the app works totally offline.
* **How are reps calculated?** We calculate joint angles (e.g., knee angle) using vector dot-products. When the angle dips below a threshold (e.g., 90 degrees for a squat) it marks an "eccentric phase", and returning above the threshold completes a valid "concentric phase" rep.
