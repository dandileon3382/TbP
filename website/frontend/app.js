// ════════════════════════════════════════════════════════
//  AI Form Coach — app.js v200
// ════════════════════════════════════════════════════════

// ── DOM ────────────────────────────────────────────────────────────────────
const video          = document.getElementById('webcam');
const landmarkCanvas = document.getElementById('landmarkCanvas');
const landmarkCtx    = landmarkCanvas.getContext('2d');
const captureCanvas  = document.getElementById('captureCanvas');
const captureCtx     = captureCanvas.getContext('2d');

const startBtn        = document.getElementById('startBtn');
const stopBtn         = document.getElementById('stopBtn');
const exerciseSelect  = document.getElementById('exerciseSelect');
const repCountUI      = document.getElementById('repCount');
const feedbackTextUI  = document.getElementById('feedbackText');
const debugText       = document.getElementById('debugText');
const analysisSection = document.getElementById('analysisSection');
const finalContent    = document.getElementById('finalAnalysisContent');
const downloadLink    = document.getElementById('downloadLink');
const spaceHint       = document.getElementById('spaceHint');

const tutorialModal   = document.getElementById('tutorialModal');
const tutorialVideo   = document.getElementById('tutorialVideo');
const readyBtn        = document.getElementById('readyBtn');
const skipBtn         = document.getElementById('skipBtn');

const pauseOverlay    = document.getElementById('pauseOverlay');

// ── Video filename map ─────────────────────────────────────────────────────
// Maps exercise key → actual filename in /videos/ directory
const VIDEO_MAP = {
    squats:      'squats',
    lunges:      'lunges',
    pushups:     'pushups',
    tricep_dips: 'dips',
    situps:      'situps'
};

// ── State ──────────────────────────────────────────────────────────────────
let ws               = null;
let mediaRecorder    = null;
let recordedChunks   = [];
let isSessionActive  = false;
let isPaused         = false;
let loopTimeout      = null;
let lastLandmarks    = null;
let frameCount       = 0;
let waitingForResponse = false;
let mistakesLog      = [];   // { timestamp, mistake, feedback }[]

// ── Audio / Speech ──────────────────────────────────────────────────────────
const synth = window.speechSynthesis;
let lastSpokenText = null;

// ── Skeleton connections ────────────────────────────────────────────────────
const SKELETON_CONNECTIONS = [
    ['left_shoulder',  'right_shoulder'],
    ['left_shoulder',  'left_elbow'],   ['left_elbow',  'left_wrist'],
    ['right_shoulder', 'right_elbow'],  ['right_elbow', 'right_wrist'],
    ['left_shoulder',  'left_hip'],     ['right_shoulder', 'right_hip'],
    ['left_hip',       'right_hip'],
    ['left_hip',       'left_knee'],    ['left_knee',  'left_ankle'],
    ['right_hip',      'right_knee'],   ['right_knee', 'right_ankle'],
];

// ════════════════════════════════════════════════════════
//  Webcam Init
// ════════════════════════════════════════════════════════
async function initWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
            audio: false
        });
        video.srcObject = stream;

        await new Promise(resolve => {
            video.onloadedmetadata = () => { video.play(); resolve(); };
        });

        debugText.textContent = `Camera: ${video.videoWidth}×${video.videoHeight}`;

        try {
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
            mediaRecorder.onstop = saveVideo;
        } catch (err) {
            console.warn('MediaRecorder unavailable:', err);
        }
    } catch (e) {
        feedbackTextUI.textContent = 'Camera access denied — please allow camera.';
        debugText.textContent = `Error: ${e.message}`;
    }
}

// ════════════════════════════════════════════════════════
//  Speech
// ════════════════════════════════════════════════════════
function speak(text) {
    if (!text || synth.speaking || lastSpokenText === text) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.05;
    utter.pitch = 1.0;
    synth.speak(utter);
    lastSpokenText = text;
    setTimeout(() => { lastSpokenText = null; }, 5000);
}

// ════════════════════════════════════════════════════════
//  Tutorial Modal
// ════════════════════════════════════════════════════════
function showTutorial() {
    const ex = exerciseSelect.value;
    const filename = VIDEO_MAP[ex] || ex;

    tutorialVideo.src = `/videos/${filename}.mp4`;
    tutorialModal.classList.remove('hidden');

    tutorialVideo.play().catch(() => {
        // Auto-play blocked — user will press play manually
    });

    tutorialVideo.onended = () => closeTutorialAndStart();
}

function closeTutorialAndStart() {
    tutorialModal.classList.add('hidden');
    tutorialVideo.pause();
    tutorialVideo.src = '';
    tutorialVideo.onended = null;
    startSession();
}

// ════════════════════════════════════════════════════════
//  Spacebar Pause / Resume
// ════════════════════════════════════════════════════════
document.addEventListener('keydown', e => {
    if (e.code !== 'Space') return;
    // Prevent page scroll during a session
    if (isSessionActive) e.preventDefault();
    if (!isSessionActive) return;

    isPaused = !isPaused;

    if (isPaused) {
        pauseOverlay.classList.remove('hidden');
        spaceHint.classList.remove('visible');
        synth.cancel();
    } else {
        pauseOverlay.classList.add('hidden');
        spaceHint.classList.add('visible');
        // Restart the frame loop
        sendFrameLoop();
    }
});

// ════════════════════════════════════════════════════════
//  Session Control
// ════════════════════════════════════════════════════════
function startSession() {
    isSessionActive  = true;
    isPaused         = false;
    frameCount       = 0;
    lastLandmarks    = null;
    recordedChunks   = [];
    mistakesLog      = [];
    waitingForResponse = false;

    repCountUI.textContent    = '0';
    feedbackTextUI.textContent = 'Connecting...';
    feedbackTextUI.style.color = '';

    spaceHint.classList.add('visible');

    if (mediaRecorder && mediaRecorder.state === 'inactive') mediaRecorder.start();

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${protocol}://${window.location.host}/ws/form_detection`);

    ws.onopen = () => {
        debugText.textContent = 'Connected — setting exercise...';
        ws.send(JSON.stringify({ type: 'set_exercise', exercise: exerciseSelect.value }));
    };

    ws.onmessage = event => {
        waitingForResponse = false;
        const data = JSON.parse(event.data);

        // Acknowledgement after exercise set
        if (data.status) {
            debugText.textContent = `Exercise: ${data.exercise} — session active`;
            sendFrameLoop();
            return;
        }

        // Rep count
        if (data.rep_count !== undefined) repCountUI.textContent = data.rep_count;

        // Feedback
        if (data.mistake) {
            feedbackTextUI.textContent = data.feedback;
            feedbackTextUI.style.color = 'var(--alert)';
            speak(data.feedback);

            const last = mistakesLog[mistakesLog.length - 1];
            if (!last || last.mistake !== data.mistake || (Date.now() / 1000 - last.timestamp) > 3) {
                mistakesLog.push({
                    timestamp: Math.round(Date.now() / 1000),
                    mistake:   data.mistake,
                    feedback:  data.feedback
                });
            }
        } else {
            feedbackTextUI.textContent = data.feedback || 'Good form ✓';
            feedbackTextUI.style.color = 'var(--teal)';
        }

        if (data.landmarks) {
            lastLandmarks = data.landmarks;
            drawLandmarks(data.landmarks);
        }

        debugText.textContent = `Frame #${frameCount} | Reps: ${data.rep_count} | Issues: ${mistakesLog.length}`;
    };

    ws.onerror = () => { debugText.textContent = 'WebSocket error!'; };
    ws.onclose = () => { console.log('WS closed'); };

    startBtn.disabled = true;
    stopBtn.disabled  = false;
    exerciseSelect.disabled = true;
    analysisSection.classList.add('hidden');
}

function stopSession() {
    isSessionActive = false;
    isPaused        = false;
    clearTimeout(loopTimeout);
    pauseOverlay.classList.add('hidden');
    spaceHint.classList.remove('visible');

    if (ws) { ws.close(); ws = null; }
    if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();

    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

    startBtn.disabled        = false;
    stopBtn.disabled         = true;
    exerciseSelect.disabled  = false;

    analysisSection.classList.remove('hidden');
    renderSessionSummary();
}

// ════════════════════════════════════════════════════════
//  Session Summary (pre-RAG)
// ════════════════════════════════════════════════════════
function renderSessionSummary() {
    const ex          = exerciseSelect.value;
    const exLabel     = exerciseSelect.options[exerciseSelect.selectedIndex].text;
    const totalReps   = parseInt(repCountUI.textContent) || 0;
    const totalIssues = mistakesLog.length;

    finalContent.innerHTML = `
        <div class="summary-header">
            <div class="summary-stat">
                <span class="stat-label">Exercise</span>
                <span class="stat-value plain" style="font-size:1.1rem;line-height:1.4">${exLabel}</span>
            </div>
            <div class="summary-stat">
                <span class="stat-label">Reps</span>
                <span class="stat-value accent">${totalReps}</span>
            </div>
            <div class="summary-stat">
                <span class="stat-label">Form Issues</span>
                <span class="stat-value ${totalIssues > 0 ? 'alert' : 'success'}">${totalIssues}</span>
            </div>
        </div>

        ${totalIssues > 0 ? `
        <div class="mistakes-log">
            <h3>Issues Detected This Session</h3>
            ${mistakesLog.map(m => `
                <div class="mistake-item">
                    <span class="mistake-time">${formatTime(m.timestamp)}</span>
                    <span class="mistake-text">${m.feedback}</span>
                </div>
            `).join('')}
        </div>
        ` : `<p class="no-mistakes">✅ Excellent session — no form issues detected. Your technique looked solid!</p>`}

        <button id="deepAnalysisBtn" class="deep-btn">
            🧠 Get AI Coaching Report
        </button>
        <div id="ragResult" class="rag-result hidden">
            <div class="rag-loading">Analysing your session with Coach Alex...</div>
        </div>
    `;

    document.getElementById('deepAnalysisBtn').addEventListener('click', runDeepAnalysis);
}

// ── Sensor HUD ──────────────────────────────────────────────────────────────
async function pollSensors() {
    if (!isSessionActive || isPaused) return;
    try {
        const res = await fetch('/sensor_data');
        const data = await res.json();
        
        const tiltUI = document.getElementById('waistTilt');
        const waistStatus = document.getElementById('waistStatus');

        if (data.waist) {
            tiltUI.textContent = Math.round(data.waist.tilt || 0);
            waistStatus.textContent = 'Active';
            waistStatus.style.color = 'var(--teal)';
        }
    } catch (e) {
        console.error("Sensor poll error:", e);
    }
}

// ── Chat Widget ──────────────────────────────────────────────────────────────
const chatInput = document.getElementById('chatInput');
const chatSend = document.getElementById('chatSend');
const chatMessages = document.getElementById('chatMessages');
const chatWidget = document.getElementById('trainerChat');
const chatToggle = document.getElementById('chatToggle');

chatToggle.onclick = () => chatWidget.classList.toggle('collapsed');

async function sendChat() {
    const msg = chatInput.value.trim();
    if (!msg) return;

    appendMessage(msg, 'user');
    chatInput.value = '';
    chatSend.disabled = true;

    // Show a loading bubble
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'msg trainer';
    loadingDiv.id = 'chatLoading';
    loadingDiv.textContent = 'Thinking…';
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                session_context: `Exercise: ${exerciseSelect.value}. Reps: ${repCountUI.textContent}. Form issues: ${mistakesLog.length}.`
            })
        });

        document.getElementById('chatLoading')?.remove();

        if (!res.ok) {
            appendMessage("The server had an issue — please try again.", 'trainer');
            return;
        }

        const data = await res.json();
        const reply = data.response;
        if (!reply) {
            appendMessage("I didn't catch a response — please try again.", 'trainer');
            return;
        }
        appendMessage(reply, 'trainer');

    } catch (e) {
        document.getElementById('chatLoading')?.remove();
        appendMessage("Sorry, I'm having trouble connecting right now.", 'trainer');
    } finally {
        chatSend.disabled = false;
    }
}

function appendMessage(text, side) {
    const div = document.createElement('div');
    div.className = `msg ${side}`;
    if (side === 'trainer') {
        div.innerHTML = formatTrainerMsg(text);
    } else {
        div.textContent = text;
    }
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatTrainerMsg(text) {
    if (!text || typeof text !== 'string') return 'Let me think about that…';
    return text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

chatSend.onclick = sendChat;
chatInput.onkeydown = (e) => { if (e.key === 'Enter') sendChat(); };

// ── Refactored Final Analysis ────────────────────────────────────────────────
async function runDeepAnalysis() {
    const btn       = document.getElementById('deepAnalysisBtn');
    const ragResult = document.getElementById('ragResult');
    const ex        = exerciseSelect.value;

    btn.disabled     = true;
    btn.textContent  = '⏳ Trainer is reviewing your session…';
    ragResult.classList.remove('hidden');
    ragResult.innerHTML = '<div class="rag-loading">🤔 Analyzing form + biometric data…</div>';

    try {
        const res  = await fetch('/final_analysis', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ exercise: ex, mistakes: mistakesLog })
        });
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();
        btn.textContent = '✅ Analysis Complete';

        // ── Form Score ──
        const score     = parseInt(data.form_score) || 0;
        const scoreColor = score >= 80 ? 'var(--teal)' : score >= 60 ? 'var(--warn)' : 'var(--alert)';
        const scoreLabel = score >= 85 ? 'Excellent' : score >= 70 ? 'Good' : score >= 50 ? 'Needs Work' : 'Poor';

        // ── Per-mistake cards ──
        const mistakeCards = (data.per_mistake_analysis || []).map(m => `
            <div class="trainer-card">
                <div class="trainer-card-header">
                    <span class="trainer-card-mistake">⚠ ${escapeHtml((m.mistake||'').replace(/_/g,' '))}</span>
                    <span class="trainer-card-count">${m.count || 0}\u00d7 detected</span>
                </div>
                <div class="trainer-card-body">
                    <div class="trainer-section">
                        <span class="trainer-section-label">Root Cause</span>
                        <div class="trainer-section-body">${escapeHtml(m.cause||'')}</div>
                    </div>
                    <div class="trainer-section">
                        <span class="trainer-section-label">Why It Matters</span>
                        <div class="trainer-section-body">${escapeHtml(m.effect||'')}</div>
                    </div>
                    <div class="trainer-section trainer-fix">
                        <span class="trainer-section-label">🎯 Corrective Drill</span>
                        <div class="trainer-section-body">${escapeHtml(m.drill||'')}</div>
                    </div>
                </div>
            </div>`).join('');

        ragResult.innerHTML = `
            <div class="form-score-block">
                <div class="score-circle" style="border-color:${scoreColor};color:${scoreColor}">
                    <span class="score-num">${score}</span>
                    <span class="score-sublabel">${scoreLabel}</span>
                </div>
                <div class="score-caption">Overall Form Score</div>
            </div>

            <div class="rag-summary-block">
                <strong>${escapeHtml(data.headline||'')}</strong><br><br>
                ${escapeHtml(data.narrative||'')}
            </div>

            ${ data.sensor_analysis ? `
            <div class="sensor-analysis-block">
                <span class="trainer-section-label">🔷 Sensor / Biometrics</span>
                <div class="trainer-section-body" style="margin-top:6px">${escapeHtml(data.sensor_analysis)}</div>
            </div>` : '' }

            ${ mistakeCards ? `
            <div class="rag-cards">
                <h3 class="section-heading">Mistake Breakdown</h3>
                ${mistakeCards}
            </div>` : '' }

            <div class="trainer-card" style="margin-top:18px;border-color:rgba(187,134,252,0.3)">
                <div class="trainer-card-header">
                    <span class="trainer-card-mistake">🎯 Action Plan for Next Session</span>
                </div>
                <div class="trainer-card-body">
                    <div class="trainer-section-body">${escapeHtml(data.action_plan||'')}</div>
                    ${ data.warmup_recommendation ? `
                    <div class="trainer-fix" style="margin-top:12px">
                        <span class="trainer-section-label">🔥 Recommended Warm-Up</span>
                        <div class="trainer-section-body" style="margin-top:4px">${escapeHtml(data.warmup_recommendation)}</div>
                    </div>` : '' }
                </div>
            </div>

            <div class="trainer-encouragement" style="margin-top:18px;font-size:1rem;padding:12px 0">
                ${escapeHtml(data.encouragement||'')}
            </div>`;

    } catch (err) {
        btn.textContent = '❌ Error — try again';
        btn.disabled = false;
        ragResult.innerHTML = `<div class="rag-loading" style="color:var(--alert)">Error: ${err.message}</div>`;
    }
}

// ── Video Loop Update ───────────────────────────────────────────────────────
function sendFrameLoop() {
    if (!isSessionActive || isPaused) return;

    if (waitingForResponse) {
        loopTimeout = setTimeout(sendFrameLoop, 60);
        return;
    }

    if (!video.videoWidth || !video.videoHeight) {
        loopTimeout = setTimeout(sendFrameLoop, 200);
        return;
    }

    captureCanvas.width  = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    const frameData = captureCanvas.toDataURL('image/jpeg', 0.6);

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ image: frameData }));
        waitingForResponse = true;
        frameCount++;
        pollSensors(); // Poll sensors on every frame or every few frames
    }

    loopTimeout = setTimeout(sendFrameLoop, 120);  // ~8 fps
}

function drawLandmarks(landmarks) {
    // Size the canvas to the exact CSS pixel dimensions of the video element
    const videoEl   = video;
    const canvasEl  = landmarkCanvas;
    const cssW      = videoEl.offsetWidth;
    const cssH      = videoEl.offsetHeight;

    if (!cssW || !cssH) return; // layout not ready yet

    // Only resize if needed (avoids flicker)
    if (canvasEl.width !== cssW || canvasEl.height !== cssH) {
        canvasEl.width  = cssW;
        canvasEl.height = cssH;
    }

    const ctx = landmarkCtx;
    ctx.clearRect(0, 0, cssW, cssH);

    // ── object-fit: cover letterbox/pillarbox compensation ────────────────────
    // MediaPipe returns coords normalised to the *original video frame* (0-1).
    // When the video CSS uses object-fit:cover the rendered image is cropped/scaled
    // so we must compute the same crop to map those normalised coords to canvas px.
    const vidW = videoEl.videoWidth  || cssW;
    const vidH = videoEl.videoHeight || cssH;

    const scale  = Math.max(cssW / vidW, cssH / vidH);
    const scaledW = vidW * scale;
    const scaledH = vidH * scale;
    const offsetX = (cssW - scaledW) / 2;
    const offsetY = (cssH - scaledH) / 2;

    // Convert a normalised landmark to canvas px
    const px = (lm) => ({
        x: lm.x * scaledW + offsetX,
        y: lm.y * scaledH + offsetY,
    });

    // ── Draw skeleton bones ───────────────────────────────────────────────────
    ctx.strokeStyle = 'rgba(3, 218, 198, 0.65)';
    ctx.lineWidth   = 2.5;
    ctx.lineCap     = 'round';

    for (const [fromName, toName] of SKELETON_CONNECTIONS) {
        const a = landmarks[fromName];
        const b = landmarks[toName];
        if (a && b && (a.visibility ?? 1) > 0.25 && (b.visibility ?? 1) > 0.25) {
            const pa = px(a);
            const pb = px(b);
            ctx.beginPath();
            ctx.moveTo(pa.x, pa.y);
            ctx.lineTo(pb.x, pb.y);
            ctx.stroke();
        }
    }

    // ── Draw joint dots ───────────────────────────────────────────────────────
    for (const lm of Object.values(landmarks)) {
        if (lm && (lm.visibility ?? 1) > 0.25) {
            const { x, y } = px(lm);
            // Outer glow
            ctx.fillStyle = 'rgba(187,134,252,0.22)';
            ctx.beginPath();
            ctx.arc(x, y, 9, 0, 2 * Math.PI);
            ctx.fill();
            // Inner dot
            ctx.fillStyle = '#bb86fc';
            ctx.beginPath();
            ctx.arc(x, y, 4.5, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

// ════════════════════════════════════════════════════════
//  Save Video Recording
// ════════════════════════════════════════════════════════
function saveVideo() {
    if (recordedChunks.length === 0) return;
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url  = URL.createObjectURL(blob);
    downloadLink.href     = url;
    downloadLink.download = `session_${Date.now()}.webm`;
    downloadLink.classList.remove('hidden');
    downloadLink.textContent = '⬇ Download Session Recording';
}

// ════════════════════════════════════════════════════════
//  Utilities
// ════════════════════════════════════════════════════════
function formatTime(ts) {
    return new Date(ts * 1000).toLocaleTimeString();
}

function escapeHtml(str) {
    if (typeof str !== 'string') return String(str);
    return str
        .replace(/&/g,  '&amp;')
        .replace(/</g,  '&lt;')
        .replace(/>/g,  '&gt;')
        .replace(/"/g,  '&quot;')
        .replace(/'/g,  '&#39;')
        .replace(/\n/g, '<br>');
}

// ════════════════════════════════════════════════════════
//  Event Listeners
// ════════════════════════════════════════════════════════
startBtn.addEventListener('click', showTutorial);
readyBtn.addEventListener('click', closeTutorialAndStart);
skipBtn.addEventListener('click',  closeTutorialAndStart);
stopBtn.addEventListener('click',  stopSession);

// ════════════════════════════════════════════════════════
//  Boot
// ════════════════════════════════════════════════════════
chatWidget.classList.add('collapsed');
initWebcam();
console.log('AI Form Coach v200 loaded');
