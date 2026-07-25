'use strict';

const API_URL       = window.appInfo.apiUrl;
const POLL_INTERVAL = 800;
const MAX_ATTEMPTS  = 40;

window.onerror = (msg, src, line, col, err) => {
    console.error('[GLOBAL ERROR]', msg, '\nat', src, 'line', line, err);
};
window.addEventListener('unhandledrejection', (e) => {
    console.error('[UNHANDLED PROMISE]', e.reason);
});

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile     = null;
let originalAudioUrl = null;
let enhancedAudioUrl = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
let startupScreen, startupStatus, appScreen, apiDot, apiText;
let viewUpload, viewFile, viewResults;
let dropZone, fileInput, fileNameEl, fileMetaEl;
let originalPlayer, originalWaveform, enhanceBtn, changeFileBtn;
let resultOriginalPlayer, resultOrigWaveform;
let enhancedPlayer, enhancedWaveform;
let downloadBtn, enhanceAnotherBtn;
let loadingOverlay, loadingStatus, toastContainer;
let analysisPlot, plotLoading;

function initDomRefs() {
    const get = (id) => {
        const el = document.getElementById(id);
        if (!el) console.error(`[DOM] Missing element: #${id}`);
        return el;
    };
    startupScreen        = get('startup-screen');
    startupStatus        = get('startup-status');
    appScreen            = get('app-screen');
    apiDot               = get('api-dot');
    apiText              = get('api-text');
    viewUpload           = get('view-upload');
    viewFile             = get('view-file');
    viewResults          = get('view-results');
    dropZone             = get('drop-zone');
    fileInput            = get('file-input');
    fileNameEl           = get('file-name');
    fileMetaEl           = get('file-meta');
    originalPlayer       = get('original-player');
    originalWaveform     = get('original-waveform');
    enhanceBtn           = get('enhance-btn');
    changeFileBtn        = get('change-file-btn');
    resultOriginalPlayer = get('result-original-player');
    resultOrigWaveform   = get('result-original-waveform');
    enhancedPlayer       = get('enhanced-player');
    enhancedWaveform     = get('enhanced-waveform');
    downloadBtn          = get('download-btn');
    enhanceAnotherBtn    = get('enhance-another-btn');
    loadingOverlay       = get('loading-overlay');
    loadingStatus        = get('loading-status');
    toastContainer       = get('toast-container');
    analysisPlot         = get('analysis-plot');
    plotLoading          = get('plot-loading');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function show(el) { if (el) el.classList.remove('hidden'); }
function hide(el) { if (el) el.classList.add('hidden'); }

function setView(name) {
    hide(viewUpload); hide(viewFile); hide(viewResults);
    const map = { upload: viewUpload, file: viewFile, results: viewResults };
    const el  = map[name];
    if (!el) return;
    show(el);
    el.classList.remove('fade-in', 'slide-up');
    void el.offsetWidth;
    el.classList.add(name === 'upload' ? 'fade-in' : 'slide-up');
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(type, message, duration = 4500) {
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const iconSpan = document.createElement('span');
    iconSpan.textContent = icons[type] || 'ℹ️';
    const msgSpan = document.createElement('span');
    msgSpan.textContent = message;
    toast.appendChild(iconSpan);
    toast.appendChild(msgSpan);
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('hiding');
        setTimeout(() => toast.remove(), 280);
    }, duration);
}

function showLoading(msg) {
    loadingStatus.textContent = msg || 'Processing…';
    show(loadingOverlay);
}
function hideLoading() { hide(loadingOverlay); }

// ── API health poll ───────────────────────────────────────────────────────────
async function pollUntilReady() {
    for (let i = 0; i < MAX_ATTEMPTS; i++) {
        try {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 2000);
            const res   = await fetch(`${API_URL}/health`, { signal: controller.signal });
            clearTimeout(timer);
            if (res.ok) return true;
        } catch (_) {}
        startupStatus.textContent = `Connecting… (${i + 1}/${MAX_ATTEMPTS})`;
        await new Promise(r => setTimeout(r, POLL_INTERVAL));
    }
    return false;
}

async function initApp() {
    const ready = await pollUntilReady();
    if (!ready) {
        startupStatus.textContent = '❌ Could not connect — start the Flask API first (python3 api.py)';
        showToast('error', 'Cannot reach enhancement engine on port 5055.');
        return;
    }
    startupScreen.style.transition = 'opacity 0.4s ease';
    startupScreen.style.opacity    = '0';
    setTimeout(() => hide(startupScreen), 420);
    show(appScreen);
    apiDot.className    = 'w-1.5 h-1.5 rounded-full bg-emerald-400';
    apiText.textContent = 'Ready';
    apiText.className   = 'text-xs text-emerald-400';
    showToast('info', 'Enhancement engine ready');
}

// ── Waveform ──────────────────────────────────────────────────────────────────
function parseWavSamples(arrayBuffer) {
    try {
        const view = new DataView(arrayBuffer);
        let offset = 12;
        while (offset + 8 < view.byteLength) {
            const id   = String.fromCharCode(
                view.getUint8(offset), view.getUint8(offset + 1),
                view.getUint8(offset + 2), view.getUint8(offset + 3)
            );
            const size = view.getUint32(offset + 4, true);
            if (id === 'data') {
                const raw    = new Int16Array(arrayBuffer, offset + 8, Math.floor(size / 2));
                const floats = new Float32Array(raw.length);
                for (let i = 0; i < raw.length; i++) floats[i] = raw[i] / 32768;
                return floats;
            }
            offset += 8 + size + (size & 1);
        }
    } catch (e) { console.error('[parseWavSamples]', e); }
    return null;
}

function drawWaveform(canvas, samples, color) {
    if (!canvas || !samples || samples.length === 0) return;
    const dpr  = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const cssH = parseInt(canvas.getAttribute('height') || '72');
    if (rect.width === 0) return;

    canvas.width        = Math.round(rect.width * dpr);
    canvas.height       = Math.round(cssH * dpr);
    canvas.style.height = cssH + 'px';

    const ctx  = canvas.getContext('2d');
    const W    = canvas.width;
    const H    = canvas.height;
    const step = Math.max(1, Math.ceil(samples.length / W));
    const amp  = H / 2;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0f1a';
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(0, amp); ctx.lineTo(W, amp); ctx.stroke();

    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0,   (color || '#00d4ff') + '55');
    grad.addColorStop(0.5, color || '#00d4ff');
    grad.addColorStop(1,   (color || '#00d4ff') + '55');
    ctx.strokeStyle = grad;
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    for (let i = 0; i < W; i++) {
        let mn = 1, mx = -1;
        for (let j = 0; j < step; j++) {
            const v = samples[i * step + j];
            if (v !== undefined) { if (v < mn) mn = v; if (v > mx) mx = v; }
        }
        ctx.moveTo(i, (1 + mn) * amp);
        ctx.lineTo(i, (1 + mx) * amp);
    }
    ctx.stroke();
}

function tryDrawWaveform(canvas, arrayBuffer, color) {
    try {
        const samples = parseWavSamples(arrayBuffer);
        if (samples) drawWaveform(canvas, samples, color);
        else if (canvas) canvas.style.display = 'none';
    } catch (e) {
        if (canvas) canvas.style.display = 'none';
    }
}

// ── Metrics ───────────────────────────────────────────────────────────────────
// renderMetricsInto is shared with live.js (id prefix 'live-') for the Live tab's
// metrics panel — both files are classic scripts in the same global scope.
function renderMetricsInto(prefix, metrics) {
    if (!metrics) return;
    const { before, after } = metrics;

    function fmt(val, unit) {
        if (val === null || val === undefined) return '—';
        return unit === 'dB' ? `${val} dB` : String(val);
    }

    function delta(b, a, unit) {
        if (b === null || b === undefined || a === null || a === undefined) return '—';
        const diff = (a - b).toFixed(unit === 'dB' ? 2 : 3);
        const sign = diff >= 0 ? '+' : '';
        const el   = document.createElement('span');
        el.className = diff >= 0 ? 'delta-positive' : 'delta-negative';
        el.textContent = `${sign}${diff}${unit === 'dB' ? ' dB' : ''} ↑`;
        return el;
    }

    function setMetric(name, b, a, unit) {
        document.getElementById(`${prefix}${name}-before`).textContent = fmt(b, unit);
        document.getElementById(`${prefix}${name}-after`).textContent  = fmt(a, unit);
        const deltaEl = document.getElementById(`${prefix}${name}-delta`);
        deltaEl.innerHTML = '';
        deltaEl.appendChild(delta(b, a, unit));
    }

    setMetric('snr',  before?.snr_db, after?.snr_db, 'dB');
    setMetric('pesq', before?.pesq,   after?.pesq,   '');
    setMetric('stoi', before?.stoi,   after?.stoi,   '');
}

function renderMetrics(metrics) { renderMetricsInto('', metrics); }

function resetMetricsDisplay(prefix) {
    for (const name of ['snr', 'pesq', 'stoi']) {
        const beforeEl = document.getElementById(`${prefix}${name}-before`);
        const afterEl  = document.getElementById(`${prefix}${name}-after`);
        const deltaEl  = document.getElementById(`${prefix}${name}-delta`);
        if (beforeEl) beforeEl.textContent = '—';
        if (afterEl)  afterEl.textContent  = '—';
        if (deltaEl)  deltaEl.textContent  = '—';
    }
}

function renderPlot(plotUrl) {
    if (!plotUrl) { hide(plotLoading); return; }
    analysisPlot.onload  = () => { show(analysisPlot); hide(plotLoading); };
    analysisPlot.onerror = () => { hide(plotLoading); };
    analysisPlot.src = plotUrl;
}

// ── File handling ─────────────────────────────────────────────────────────────
async function handleFile(file) {
    if (!file) return;
    const ext   = '.' + file.name.split('.').pop().toLowerCase();
    const valid = ['.wav', '.mp3', '.flac', '.ogg', '.m4a'];
    if (!valid.includes(ext)) { showToast('error', `Unsupported format: ${ext.toUpperCase()}`); return; }
    if (file.size > 100 * 1024 * 1024) { showToast('error', 'File too large (max 100 MB).'); return; }

    selectedFile = file;
    const sizeMB  = (file.size / 1024 / 1024).toFixed(2);
    const blobUrl = URL.createObjectURL(file);

    fileNameEl.textContent = file.name;
    fileMetaEl.textContent = `${sizeMB} MB`;
    originalPlayer.src     = blobUrl;

    originalPlayer.addEventListener('loadedmetadata', () => {
        const dur = isFinite(originalPlayer.duration)
            ? ` · ${originalPlayer.duration.toFixed(1)}s` : '';
        fileMetaEl.textContent = `${sizeMB} MB${dur}`;
    }, { once: true });

    setView('file');
}

// ── Enhancement ───────────────────────────────────────────────────────────────
async function enhance() {
    if (!selectedFile) return;
    showLoading('Sending to enhancement engine…');
    enhanceBtn.disabled = true;

    const msgs = [
        'Analyzing noise profile…',
        'Stage 1: Spectral gating…',
        'Stage 2: DeepFilterNet…',
        'Computing metrics…',
    ];
    let msgIdx = 0;
    const msgTimer = setInterval(() => {
        msgIdx = (msgIdx + 1) % msgs.length;
        try { loadingStatus.textContent = msgs[msgIdx]; } catch (_) {}
    }, 1400);

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const res = await fetch(`${API_URL}/denoise`, { method: 'POST', body: formData });
        clearInterval(msgTimer);

        let data;
        try { data = await res.json(); } catch (_) { throw new Error('API returned non-JSON response'); }
        if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
        if (!data.original_url || !data.enhanced_url) throw new Error('API response missing audio URLs');

        originalAudioUrl = data.original_url;
        enhancedAudioUrl = data.enhanced_url;

        resultOriginalPlayer.src = originalAudioUrl;
        enhancedPlayer.src       = enhancedAudioUrl;

        // Reset plot state
        hide(analysisPlot);
        show(plotLoading);

        setView('results');
        showToast('success', 'Audio enhanced successfully!');

        // Render metrics
        renderMetrics(data.metrics);

        // Render plot
        renderPlot(data.plot_url);

        // Waveforms
        Promise.all([
            fetch(originalAudioUrl).then(r => r.arrayBuffer()),
            fetch(enhancedAudioUrl).then(r => r.arrayBuffer()),
        ]).then(([origAb, enhAb]) => {
            tryDrawWaveform(resultOrigWaveform, origAb, '#475569');
            tryDrawWaveform(enhancedWaveform,   enhAb,  '#00d4ff');
        }).catch(e => console.error('[Waveform fetch]', e));

    } catch (e) {
        clearInterval(msgTimer);
        showToast('error', e.message || 'Enhancement failed');
    } finally {
        hideLoading();
        enhanceBtn.disabled = false;
    }
}

// ── Download ──────────────────────────────────────────────────────────────────
function downloadEnhanced() {
    if (!enhancedAudioUrl || !selectedFile) return;
    const base = selectedFile.name.replace(/\.[^.]+$/, '');
    const a    = document.createElement('a');
    a.href     = enhancedAudioUrl;
    a.download = `enhanced_${base}.wav`;
    a.click();
    showToast('success', 'Download started!');
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function reset() {
    selectedFile = null; originalAudioUrl = null; enhancedAudioUrl = null;
    originalPlayer.src       = '';
    resultOriginalPlayer.src = '';
    enhancedPlayer.src       = '';
    fileInput.value          = '';
    analysisPlot.src         = '';
    hide(analysisPlot);
    setView('upload');
}

// ── Events ────────────────────────────────────────────────────────────────────
function bindEvents() {
    dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('drop-active'); });
    dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drop-active'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drop-active');
        const f = e.dataTransfer.files[0];
        if (f) handleFile(f).catch(console.error);
    });
    fileInput.addEventListener('change', (e) => {
        const f = e.target.files[0];
        if (f) handleFile(f).catch(console.error);
    });
    enhanceBtn.addEventListener('click',        () => enhance().catch(console.error));
    changeFileBtn.addEventListener('click',     reset);
    downloadBtn.addEventListener('click',       downloadEnhanced);
    enhanceAnotherBtn.addEventListener('click', reset);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    initDomRefs();
    bindEvents();
    initApp().catch(console.error);
});
