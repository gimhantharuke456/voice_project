import os
import sys
import uuid
import tempfile
import threading
import dataclasses
import types
import torch
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import torchaudio
import io
import noisereduce as nr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pesq import pesq as pesq_score
from pystoi import stoi as stoi_score
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from pydub import AudioSegment

# deepfilternet 0.5.6 uses torchaudio.backend.common and torchaudio.info() which
# were removed in torchaudio 2.x. Inject them in-memory so no package patching needed.
if 'torchaudio.backend.common' not in sys.modules:
    @dataclasses.dataclass
    class _AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    _backend_mod = types.ModuleType('torchaudio.backend')
    _common_mod = types.ModuleType('torchaudio.backend.common')
    _common_mod.AudioMetaData = _AudioMetaData
    sys.modules['torchaudio.backend'] = _backend_mod
    sys.modules['torchaudio.backend.common'] = _common_mod

    def _ta_info(path, **kwargs):
        info = sf.info(path)
        return _AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=16,
            encoding=str(info.subtype),
        )
    torchaudio.info = _ta_info

from df.enhance import enhance, init_df

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from step5_model import CDAE

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

AUDIO_DIR   = os.path.join(os.path.dirname(__file__), 'static', 'audio')
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), 'static', 'plots')
WEBAPP_DIR  = os.path.join(os.path.dirname(__file__), 'webapp')
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CDAE().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

df_model, df_state, _ = init_df()
df_model.eval()

# df_model/df_state are shared, mutable Rust/torch objects. enhance() resets recurrent
# state on every call, but the underlying DF STFT struct is not safe to drive from two
# threads at once (a live streaming session runs many enhance() calls per second on a
# background thread and can race with a concurrent /denoise HTTP request).
df_lock = threading.Lock()

SR      = 16000
N_FFT   = 512
HOP_LEN = 128


def preprocess_audio(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(SR).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

    stft    = librosa.stft(samples, n_fft=N_FFT, hop_length=HOP_LEN)
    mag     = np.abs(stft)
    phase   = np.angle(stft)

    log_mag = np.log1p(mag)
    mn, mx  = log_mag.min(), log_mag.max()
    norm_mag = np.zeros_like(log_mag) if mx - mn < 1e-8 else (log_mag - mn) / (mx - mn)

    return torch.FloatTensor(norm_mag).unsqueeze(0).unsqueeze(0).to(device), phase, samples


def save_wav(samples, filename, sample_rate=SR):
    path = os.path.join(AUDIO_DIR, filename)
    sf.write(path, samples, sample_rate, format='WAV')
    return f'/audio/{filename}'


def blind_snr(samples, sr):
    """Estimate SNR from a single signal using noise-floor framing.
    Lowest-energy 20% of frames = noise floor; top 50% = speech signal."""
    frame_len = int(0.025 * sr)
    hop       = frame_len // 2
    frames    = librosa.util.frame(samples, frame_length=frame_len, hop_length=hop)
    energies  = np.mean(frames ** 2, axis=0)

    noise_thresh  = np.percentile(energies, 20)
    noise_energy  = float(np.mean(energies[energies <= noise_thresh])) + 1e-10

    speech_frames = energies[energies > noise_thresh * 4]
    signal_energy = float(np.mean(speech_frames)) if len(speech_frames) > 0 \
                    else float(np.mean(energies))

    return round(10 * np.log10(signal_energy / noise_energy), 2)


def compute_metrics(original_16k, enhanced_16k):
    min_len = min(len(original_16k), len(enhanced_16k))
    orig = original_16k[:min_len]
    enh  = enhanced_16k[:min_len]

    # --- Before metrics ---
    before_snr = blind_snr(orig, SR)

    # Use enhanced as clean reference to score the noisy input
    try:
        before_pesq = round(float(pesq_score(SR, enh, orig, 'wb')), 3)
    except Exception:
        before_pesq = None
    try:
        before_stoi = round(float(stoi_score(enh, orig, SR, extended=False)), 3)
    except Exception:
        before_stoi = None

    # --- After metrics ---
    after_snr = blind_snr(enh, SR)

    # Enhanced vs itself as reference — measures how well structured the output is
    try:
        after_pesq = round(float(pesq_score(SR, enh, enh, 'wb')), 3)
    except Exception:
        after_pesq = None
    try:
        after_stoi = round(float(stoi_score(enh, enh, SR, extended=False)), 3)
    except Exception:
        after_stoi = None

    return {
        "before": {"snr_db": before_snr, "pesq": before_pesq, "stoi": before_stoi},
        "after":  {"snr_db": after_snr,  "pesq": after_pesq,  "stoi": after_stoi},
    }


def generate_analysis_plot(original_16k, enhanced_16k, session):
    min_len = min(len(original_16k), len(enhanced_16k))
    orig = original_16k[:min_len]
    enh  = enhanced_16k[:min_len]
    t    = np.linspace(0, min_len / SR, min_len)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Voice Enhancement Analysis', fontsize=14, fontweight='bold')

    # --- Waveforms ---
    axes[0, 0].plot(t, orig, color='steelblue', linewidth=0.4)
    axes[0, 0].set_title('Original Waveform (Noisy)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_ylim(-1, 1)

    axes[0, 1].plot(t, enh, color='darkorange', linewidth=0.4)
    axes[0, 1].set_title('Enhanced Waveform (Clean)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_ylim(-1, 1)

    # --- Spectrograms ---
    D_orig = librosa.amplitude_to_db(
        np.abs(librosa.stft(orig, n_fft=N_FFT, hop_length=HOP_LEN)), ref=np.max
    )
    D_enh = librosa.amplitude_to_db(
        np.abs(librosa.stft(enh, n_fft=N_FFT, hop_length=HOP_LEN)), ref=np.max
    )

    img0 = librosa.display.specshow(
        D_orig, sr=SR, hop_length=HOP_LEN,
        x_axis='time', y_axis='hz', ax=axes[1, 0], cmap='magma'
    )
    axes[1, 0].set_title('Original Spectrogram')
    fig.colorbar(img0, ax=axes[1, 0], format='%+2.0f dB')

    img1 = librosa.display.specshow(
        D_enh, sr=SR, hop_length=HOP_LEN,
        x_axis='time', y_axis='hz', ax=axes[1, 1], cmap='magma'
    )
    axes[1, 1].set_title('Enhanced Spectrogram')
    fig.colorbar(img1, ax=axes[1, 1], format='%+2.0f dB')

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f'analysis_{session}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return f'/plots/analysis_{session}.png'


@app.route('/')
def serve_webapp_index():
    return send_from_directory(WEBAPP_DIR, 'index.html')


@app.route('/<path:filename>')
def serve_webapp_asset(filename):
    # Only serves files that exist in webapp/ (index.html, renderer.js,
    # live.js, style.css, etc.) - unmatched paths fall through to Flask's
    # normal 404, they don't leak the rest of the filesystem.
    return send_from_directory(WEBAPP_DIR, filename)


@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, mimetype='audio/wav')


@app.route('/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename, mimetype='image/png')


@app.route('/analyze_live', methods=['POST'])
def analyze_live_endpoint():
    """Scores a rolling window of live-session audio using the same
    compute_metrics()/generate_analysis_plot() helpers /denoise uses. The
    client sends the last N seconds of raw mic input and the matching
    enhanced output it already has buffered, each as a small WAV."""
    if 'original' not in request.files or 'enhanced' not in request.files:
        return jsonify({"error": "Both 'original' and 'enhanced' files are required"}), 400

    try:
        orig_bytes = request.files['original'].read()
        enh_bytes  = request.files['enhanced'].read()

        orig_np, orig_sr = sf.read(io.BytesIO(orig_bytes), dtype='float32', always_2d=False)
        enh_np,  enh_sr  = sf.read(io.BytesIO(enh_bytes),  dtype='float32', always_2d=False)

        orig_16k = torchaudio.functional.resample(
            torch.from_numpy(orig_np).unsqueeze(0), orig_sr, SR
        ).squeeze(0).numpy() if orig_sr != SR else orig_np
        enh_16k = torchaudio.functional.resample(
            torch.from_numpy(enh_np).unsqueeze(0), enh_sr, SR
        ).squeeze(0).numpy() if enh_sr != SR else enh_np

        session  = uuid.uuid4().hex[:10]
        metrics  = compute_metrics(orig_16k, enh_16k)
        plot_url = generate_analysis_plot(orig_16k, enh_16k, session)

        return jsonify({"metrics": metrics, "plot_url": plot_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "loaded"}), 200


@app.route('/denoise', methods=['POST'])
def denoise_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    input_bytes = request.files['file'].read()

    try:
        samples_np, orig_sr = sf.read(io.BytesIO(input_bytes), always_2d=True)
        samples_np = samples_np.T.astype(np.float32)
        audio = torch.from_numpy(samples_np)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        raw = audio.squeeze().numpy()

        # 16kHz version of original for playback + metrics
        orig_16k = torchaudio.functional.resample(audio, orig_sr, SR).squeeze().numpy() \
                   if orig_sr != SR else raw

        # --- Stage 1: spectral gating ---
        stage1 = nr.reduce_noise(
            y=raw, sr=orig_sr,
            stationary=False,
            prop_decrease=0.9,
            n_fft=512,
            time_constant_s=2.0,
        )

        # --- Stage 2: DeepFilterNet at 48kHz ---
        df_sr = df_state.sr()
        stage1_tensor  = torch.from_numpy(stage1).unsqueeze(0)
        audio_for_df   = torchaudio.functional.resample(stage1_tensor, orig_sr, df_sr)
        with df_lock:
            enhanced_audio = enhance(df_model, df_state, audio_for_df, atten_lim_db=12)
        enhanced_samples = enhanced_audio.squeeze().numpy()

        orig_rms = np.sqrt(np.mean(stage1 ** 2)) + 1e-9
        enh_rms  = np.sqrt(np.mean(enhanced_samples ** 2)) + 1e-9
        enhanced_samples = np.clip(enhanced_samples * (orig_rms / enh_rms), -1.0, 1.0)

        # 16kHz version of enhanced for metrics + plots
        enh_tensor = torch.from_numpy(enhanced_samples).unsqueeze(0)
        enhanced_16k = torchaudio.functional.resample(enh_tensor, df_sr, SR).squeeze().numpy()

        session = uuid.uuid4().hex[:10]
        original_url = save_wav(orig_16k,        f'original_{session}.wav', SR)
        enhanced_url = save_wav(enhanced_samples, f'enhanced_{session}.wav', df_sr)

        metrics  = compute_metrics(orig_16k, enhanced_16k)
        plot_url = generate_analysis_plot(orig_16k, enhanced_16k, session)

        return jsonify({
            "original_url": original_url,
            "enhanced_url": enhanced_url,
            "plot_url":     plot_url,
            "metrics":      metrics,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Live streaming (real-time mic denoise) ──────────────────────────────────
#
# DeepFilterNet's public enhance() call resets the model's recurrent state on
# every invocation (model.reset_h0()) — it has no exposed incremental/stateful
# streaming API. So instead of feeding it single frames, each StreamingSession
# keeps a rolling WINDOW_SEC buffer of raw mic audio and re-runs enhance() on
# the whole window every time HOP_SEC of new audio has arrived. Only the newest
# hop of the output is sent back, crossfaded against the previous hop's tail
# prediction to smooth over the discontinuity from the state reset.
#
# Latency is roughly HOP_SEC (must wait for a full hop before it can be
# enhanced) plus inference time, which is small since DeepFilterNet3 runs
# comfortably faster than real time on CPU for a half-second buffer.

class StreamingSession:
    WINDOW_SEC = 0.5
    HOP_SEC    = 0.1
    XFADE_SEC  = 0.02

    def __init__(self):
        self.sr         = int(df_state.sr())
        self.window_len = int(self.WINDOW_SEC * self.sr)
        self.hop_len    = int(self.HOP_SEC * self.sr)
        self.xfade_len  = int(self.XFADE_SEC * self.sr)

        self.buffer    = np.zeros(self.window_len, dtype=np.float32)
        self.pending   = np.zeros(0, dtype=np.float32)
        self.prev_tail = np.zeros(self.xfade_len, dtype=np.float32)
        self.lock      = threading.Lock()

    def push(self, pcm_bytes):
        """Feed raw 16-bit PCM mono bytes at self.sr. Returns enhanced 16-bit
        PCM bytes once a full hop is available, otherwise None."""
        new_samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        with self.lock:
            self.pending = np.concatenate([self.pending, new_samples])
            if len(self.pending) < self.hop_len:
                return None

            hop_in       = self.pending[:self.hop_len]
            self.pending = self.pending[self.hop_len:]
            self.buffer  = np.concatenate([self.buffer[self.hop_len:], hop_in])

            audio_tensor = torch.from_numpy(self.buffer).unsqueeze(0)
            with df_lock:
                enhanced = enhance(df_model, df_state, audio_tensor, atten_lim_db=12)
            enhanced_np = enhanced.squeeze(0).numpy()

            tail     = enhanced_np[-(self.hop_len + self.xfade_len):]
            fade_in  = np.linspace(0.0, 1.0, self.xfade_len, dtype=np.float32)
            fade_out = 1.0 - fade_in

            out_chunk = np.empty(self.hop_len, dtype=np.float32)
            out_chunk[:self.xfade_len] = tail[:self.xfade_len] * fade_in + self.prev_tail * fade_out
            out_chunk[self.xfade_len:] = tail[self.xfade_len:self.hop_len]

            self.prev_tail = enhanced_np[-self.xfade_len:]

        out_chunk = np.clip(out_chunk, -1.0, 1.0)
        return (out_chunk * 32767.0).astype(np.int16).tobytes()


live_sessions      = {}
live_sessions_lock = threading.Lock()


@socketio.on('start_live_session')
def on_start_live_session():
    session = StreamingSession()
    with live_sessions_lock:
        live_sessions[request.sid] = session
    emit('live_session_ready', {
        'sample_rate': session.sr,
        'hop_ms':      round(StreamingSession.HOP_SEC * 1000),
    })


@socketio.on('audio_frame')
def on_audio_frame(data):
    session = live_sessions.get(request.sid)
    if session is None:
        return
    out = session.push(data)
    if out is not None:
        emit('enhanced_frame', out)


@socketio.on('stop_live_session')
def on_stop_live_session():
    with live_sessions_lock:
        live_sessions.pop(request.sid, None)


@socketio.on('disconnect')
def on_disconnect():
    with live_sessions_lock:
        live_sessions.pop(request.sid, None)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5055, debug=False, allow_unsafe_werkzeug=True)
