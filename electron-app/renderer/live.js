'use strict';

// Real-time mic denoising: captures mic audio via an AudioWorklet, streams
// 100ms hops to the Flask-SocketIO backend over WebSocket, and schedules the
// returned enhanced hops for gapless playback. Independent of renderer.js's
// upload/analyze flow — shares only its `show`/`hide`/`showToast`/
// `renderMetricsInto`/`resetMetricsDisplay` helpers, which are safe to reuse
// since both files are classic (non-module) scripts in the same global scope.
//
// Metrics: DeepFilterNet has no ground-truth "clean" reference during a live
// session, so SNR/PESQ/STOI are computed the same blind way /denoise does —
// by periodically POSTing a rolling window of raw mic input + matching
// enhanced output to /analyze_live (see StreamingSession comment in api.py
// for why this can't just reuse the model's internal state).

const LIVE_API_URL = window.appInfo.apiUrl;

let tabUploadBtn, tabLiveBtn, modeUpload, modeLive;
let liveDot, liveStatus, liveToggleBtn, liveToggleLabel;
let levelInputEl, levelOutputEl, recordToggle, saveRecordingBtn;
let liveMetricsStatusEl;

let socket          = null;
let audioContext    = null;
let mediaStream     = null;
let sourceNode      = null;
let captureNode     = null;
let silentGain      = null;

let isLive        = false;
let sessionReady  = false;
let liveSampleRate = 48000;
let hopLen         = 4800;

let queueParts = [];
let queueLen   = 0;
let nextPlayTime = 0;

let isRecording    = false;
let recordedChunks = [];

const METRICS_INTERVAL_MS = 4000;
const METRICS_WINDOW_SEC  = 2.5;

let metricsOrigChunks = [];
let metricsOrigLen    = 0;
let metricsEnhChunks  = [];
let metricsEnhLen     = 0;
let metricsTimer      = null;
let metricsBusy       = false;

function initLiveDomRefs() {
    tabUploadBtn     = document.getElementById('tab-upload-btn');
    tabLiveBtn       = document.getElementById('tab-live-btn');
    modeUpload       = document.getElementById('mode-upload');
    modeLive         = document.getElementById('mode-live');
    liveDot          = document.getElementById('live-dot');
    liveStatus       = document.getElementById('live-status');
    liveToggleBtn    = document.getElementById('live-toggle-btn');
    liveToggleLabel  = document.getElementById('live-toggle-label');
    levelInputEl     = document.getElementById('level-input');
    levelOutputEl    = document.getElementById('level-output');
    recordToggle     = document.getElementById('record-toggle');
    saveRecordingBtn = document.getElementById('save-recording-btn');
    liveMetricsStatusEl = document.getElementById('live-metrics-status');
}

// ── Tabs ──────────────────────────────────────────────────────────────────
function setActiveTab(name) {
    const isUpload = name === 'upload';
    show(isUpload ? modeUpload : modeLive);
    hide(isUpload ? modeLive : modeUpload);

    const active   = 'px-3 py-1 rounded-full text-xs font-medium transition-colors bg-cyan-500/15 text-cyan-400';
    const inactive = 'px-3 py-1 rounded-full text-xs font-medium transition-colors text-slate-500 hover:text-slate-300';
    tabUploadBtn.className = isUpload ? active : inactive;
    tabLiveBtn.className   = isUpload ? inactive : active;
}

// ── DSP helpers ───────────────────────────────────────────────────────────
function computeRms(float32) {
    let sum = 0;
    for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
    return Math.sqrt(sum / float32.length);
}

function updateLevel(el, rms) {
    if (!el) return;
    el.style.width = Math.min(100, rms * 350) + '%';
}

function floatTo16BitPCM(float32) {
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        out[i] = s < 0 ? s * 32768 : s * 32767;
    }
    return out;
}

function int16ToFloat32(arrayBuffer) {
    const int16 = new Int16Array(arrayBuffer);
    const out   = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) out[i] = int16[i] / 32768;
    return out;
}

// ── Capture → hop batching → send ────────────────────────────────────────
function onCaptureChunk(chunk) {
    if (!sessionReady) return;

    queueParts.push(chunk);
    queueLen += chunk.length;
    if (queueLen < hopLen) return;

    const combined = new Float32Array(queueLen);
    let offset = 0;
    for (const part of queueParts) { combined.set(part, offset); offset += part.length; }
    queueParts = [];
    queueLen   = 0;

    let start = 0;
    while (combined.length - start >= hopLen) {
        sendHop(combined.subarray(start, start + hopLen));
        start += hopLen;
    }
    if (start < combined.length) {
        const remainder = combined.slice(start);
        queueParts = [remainder];
        queueLen   = remainder.length;
    }
}

function sendHop(float32Hop) {
    updateLevel(levelInputEl, computeRms(float32Hop));
    metricsOrigChunks.push(float32Hop.slice());
    metricsOrigLen += float32Hop.length;
    if (socket && socket.connected) {
        socket.emit('audio_frame', floatTo16BitPCM(float32Hop).buffer);
    }
}

// ── Playback ──────────────────────────────────────────────────────────────
function schedulePlayback(float32) {
    if (!audioContext) return;

    const buffer = audioContext.createBuffer(1, float32.length, audioContext.sampleRate);
    buffer.copyToChannel(float32, 0);
    const src = audioContext.createBufferSource();
    src.buffer = buffer;
    src.connect(audioContext.destination);

    const now = audioContext.currentTime;
    if (nextPlayTime < now + 0.01) nextPlayTime = now + 0.05;
    src.start(nextPlayTime);
    nextPlayTime += float32.length / audioContext.sampleRate;
}

function onEnhancedFrame(arrayBuffer) {
    const float32 = int16ToFloat32(arrayBuffer);
    updateLevel(levelOutputEl, computeRms(float32));
    if (isRecording) recordedChunks.push(float32);
    metricsEnhChunks.push(float32.slice());
    metricsEnhLen += float32.length;
    schedulePlayback(float32);
}

// ── Metrics (SNR / PESQ / STOI) ──────────────────────────────────────────────
function mergeFloat32(chunks, totalLen) {
    const out = new Float32Array(totalLen);
    let offset = 0;
    for (const c of chunks) { out.set(c, offset); offset += c.length; }
    return out;
}

function setLiveMetricsStatus(text) {
    if (liveMetricsStatusEl) liveMetricsStatusEl.textContent = text;
}

async function sendMetricsSnapshot() {
    if (metricsBusy) return;
    const windowSamples = Math.round(liveSampleRate * METRICS_WINDOW_SEC);
    if (metricsOrigLen < windowSamples || metricsEnhLen < windowSamples) return;

    const orig = mergeFloat32(metricsOrigChunks, metricsOrigLen);
    const enh  = mergeFloat32(metricsEnhChunks, metricsEnhLen);
    metricsOrigChunks = []; metricsOrigLen = 0;
    metricsEnhChunks  = []; metricsEnhLen  = 0;

    metricsBusy = true;
    setLiveMetricsStatus('Computing…');
    try {
        const formData = new FormData();
        formData.append('original', encodeWav(orig, liveSampleRate), 'original.wav');
        formData.append('enhanced', encodeWav(enh, liveSampleRate), 'enhanced.wav');

        const res  = await fetch(`${LIVE_API_URL}/analyze_live`, { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);

        renderMetricsInto('live-', data.metrics);
        setLiveMetricsStatus('Live');
    } catch (e) {
        console.error('[live metrics]', e);
        setLiveMetricsStatus('Metrics unavailable');
    } finally {
        metricsBusy = false;
    }
}

function startMetricsLoop() {
    resetMetricsDisplay('live-');
    metricsOrigChunks = []; metricsOrigLen = 0;
    metricsEnhChunks  = []; metricsEnhLen  = 0;
    setLiveMetricsStatus('Gathering audio…');
    metricsTimer = setInterval(() => { sendMetricsSnapshot().catch(console.error); }, METRICS_INTERVAL_MS);
}

function stopMetricsLoop() {
    if (metricsTimer) { clearInterval(metricsTimer); metricsTimer = null; }
    metricsOrigChunks = []; metricsOrigLen = 0;
    metricsEnhChunks  = []; metricsEnhLen  = 0;
    metricsBusy = false;
    setLiveMetricsStatus('Idle');
}

// ── Session lifecycle ────────────────────────────────────────────────────
function setLiveStatus(text, colorClass) {
    liveStatus.textContent = text;
    liveDot.className = `w-1.5 h-1.5 rounded-full ${colorClass}`;
}

async function startLive() {
    liveToggleBtn.disabled = true;
    setLiveStatus('Requesting mic…', 'bg-yellow-400 animate-pulse');

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                // Acoustic echo cancellation stays on: on a laptop, mic and
                // speaker are the same device, so the enhanced audio we play
                // back would otherwise get picked straight back up by the
                // mic and re-sent through the pipeline. AEC cancels that
                // using the browser's own reference of what it's playing
                // out. Noise suppression/AGC stay off since DeepFilterNet
                // handles denoising downstream and we don't want the
                // browser's AGC fighting our own RMS normalization.
                echoCancellation: true,
                noiseSuppression: false,
                autoGainControl: false,
            },
        });

        audioContext = new AudioContext({ sampleRate: 48000 });
        await audioContext.audioWorklet.addModule('capture-processor.js');

        sourceNode  = audioContext.createMediaStreamSource(mediaStream);
        captureNode = new AudioWorkletNode(audioContext, 'capture-processor');
        captureNode.port.onmessage = (e) => onCaptureChunk(e.data);

        // A worklet with no output connection to the destination may not get
        // pulled by the render graph — route it through a silent gain node.
        silentGain = audioContext.createGain();
        silentGain.gain.value = 0;
        sourceNode.connect(captureNode);
        captureNode.connect(silentGain);
        silentGain.connect(audioContext.destination);

        queueParts = [];
        queueLen   = 0;
        nextPlayTime = 0;
        recordedChunks = [];
        hide(saveRecordingBtn);

        setLiveStatus('Connecting…', 'bg-yellow-400 animate-pulse');
        socket = io(LIVE_API_URL, { transports: ['websocket', 'polling'] });

        socket.on('connect', () => socket.emit('start_live_session'));

        socket.on('live_session_ready', (data) => {
            liveSampleRate = data.sample_rate;
            hopLen = Math.round(liveSampleRate * (data.hop_ms / 1000));
            sessionReady = true;
            isLive = true;
            setLiveStatus('Live', 'bg-emerald-400');
            liveToggleLabel.textContent = 'Stop Live Denoise';
            liveToggleBtn.disabled = false;
            startMetricsLoop();
        });

        socket.on('enhanced_frame', onEnhancedFrame);

        socket.on('connect_error', (err) => {
            showToast('error', `Live connection failed: ${err.message || err}`);
            stopLive();
        });

        socket.on('disconnect', () => {
            if (isLive) stopLive();
        });

    } catch (e) {
        showToast('error', e.message || 'Could not start live denoise');
        stopLive();
    }
}

function stopLive() {
    if (socket) {
        try { socket.emit('stop_live_session'); } catch (_) {}
        socket.removeAllListeners();
        socket.disconnect();
        socket = null;
    }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (captureNode) { captureNode.port.onmessage = null; captureNode.disconnect(); captureNode = null; }
    if (sourceNode)  { sourceNode.disconnect(); sourceNode = null; }
    if (silentGain)  { silentGain.disconnect(); silentGain = null; }
    if (audioContext) { audioContext.close().catch(() => {}); audioContext = null; }

    isLive       = false;
    sessionReady = false;
    queueParts   = [];
    queueLen     = 0;
    nextPlayTime = 0;

    stopMetricsLoop();

    updateLevel(levelInputEl, 0);
    updateLevel(levelOutputEl, 0);
    setLiveStatus('Idle', 'bg-slate-600');
    liveToggleLabel.textContent = 'Start Live Denoise';
    liveToggleBtn.disabled = false;

    if (recordedChunks.length > 0) show(saveRecordingBtn);
}

function toggleLive() {
    if (isLive) stopLive();
    else startLive().catch(console.error);
}

// ── Recording ─────────────────────────────────────────────────────────────
function encodeWav(float32, sampleRate) {
    const numSamples = float32.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view   = new DataView(buffer);

    const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, numSamples * 2, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        view.setInt16(offset, s < 0 ? s * 32768 : s * 32767, true);
        offset += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
}

function saveRecording() {
    if (recordedChunks.length === 0) return;

    let total = 0;
    for (const c of recordedChunks) total += c.length;
    const merged = new Float32Array(total);
    let off = 0;
    for (const c of recordedChunks) { merged.set(c, off); off += c.length; }

    const blob = encodeWav(merged, liveSampleRate);
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `live_session_${Date.now()}.wav`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    showToast('success', 'Recording saved!');
}

// ── Events ────────────────────────────────────────────────────────────────
function bindLiveEvents() {
    tabUploadBtn.addEventListener('click', () => setActiveTab('upload'));
    tabLiveBtn.addEventListener('click',   () => setActiveTab('live'));
    liveToggleBtn.addEventListener('click', toggleLive);
    saveRecordingBtn.addEventListener('click', saveRecording);
    recordToggle.addEventListener('change', () => {
        isRecording = recordToggle.checked;
        if (isRecording) {
            recordedChunks = [];
            hide(saveRecordingBtn);
        }
    });
    window.addEventListener('beforeunload', () => { try { stopLive(); } catch (_) {} });
}

// ── Boot ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    initLiveDomRefs();
    bindLiveEvents();
});
