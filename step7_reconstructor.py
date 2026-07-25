"""
Step 7 — Speech Reconstructor
Usage: python3 step7_reconstructor.py --input <noisy.wav>

Pipeline: load WAV → STFT → normalise → CDAE inference → denormalise → ISTFT → save WAV
Phase reconstruction uses the noisy file's original phase (no Griffin-Lim needed).
"""

import argparse
import os
import sys
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from step5_model import CDAE

SR       = 16000
N_FFT    = 512
HOP_LEN  = 128
MODEL_PATH  = "trained/model.pth"
OUTPUT_DIR  = "outputs/enhanced"
PLOT_DIR    = "outputs/plots"

BOLD  = "\033[1m"; GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"


def load_model(path, device):
    model = CDAE().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def preprocess(audio):
    stft    = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LEN)
    mag     = np.abs(stft).astype(np.float32)
    phase   = np.angle(stft).astype(np.float32)
    log_mag = np.log1p(mag)
    mn, mx  = log_mag.min(), log_mag.max()
    norm    = (log_mag - mn) / (mx - mn + 1e-8)
    return norm, phase, mn, mx


def reconstruct(norm_spec, phase, mn, mx):
    log_mag  = norm_spec * (mx - mn) + mn
    mag      = np.expm1(log_mag)
    stft_out = mag * np.exp(1j * phase)
    return librosa.istft(stft_out, hop_length=HOP_LEN)


def save_comparison_plot(noisy, enhanced, stem):
    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    t_noisy   = np.linspace(0, len(noisy)   / SR, len(noisy))
    t_enhanced= np.linspace(0, len(enhanced) / SR, len(enhanced))

    # Waveforms
    axes[0, 0].plot(t_noisy,    noisy,    linewidth=0.5, color="steelblue")
    axes[0, 0].set_title("Noisy — waveform");    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 1].plot(t_enhanced, enhanced, linewidth=0.5, color="darkorange")
    axes[0, 1].set_title("Enhanced — waveform"); axes[0, 1].set_xlabel("Time (s)")

    # Spectrograms
    for ax, audio, title in [
        (axes[1, 0], noisy,    "Noisy — spectrogram"),
        (axes[1, 1], enhanced, "Enhanced — spectrogram"),
    ]:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LEN)), ref=np.max)
        librosa.display.specshow(D, sr=SR, hop_length=HOP_LEN, x_axis="time", y_axis="linear", ax=ax)
        ax.set_title(title)

    plt.suptitle(stem, fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, stem + "_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def enhance(input_path, model, device):
    audio, _ = librosa.load(input_path, sr=SR, mono=True)
    norm, phase, mn, mx = preprocess(audio)

    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out_norm = model(tensor).squeeze().cpu().numpy()

    enhanced = reconstruct(out_norm, phase, mn, mx)
    return audio, enhanced


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to noisy input WAV")
    parser.add_argument("--model",  default=MODEL_PATH, help="Path to model weights")
    args = parser.parse_args()

    print(f"\n{BOLD}=== Aviation Speech Enhancement — Step 7: Reconstructor ==={RESET}\n")

    if not os.path.exists(args.input):
        print(f"{RED}Input file not found: {args.input}{RESET}"); sys.exit(1)
    if not os.path.exists(args.model):
        print(f"{RED}Model not found: {args.model}{RESET}"); sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,   exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Input  : {args.input}")
    print(f"Model  : {args.model}\n")

    model = load_model(args.model, device)
    noisy_audio, enhanced_audio = enhance(args.input, model, device)

    stem     = os.path.splitext(os.path.basename(args.input))[0]
    out_wav  = os.path.join(OUTPUT_DIR, stem + "_enhanced.wav")
    sf.write(out_wav, enhanced_audio, SR, subtype="PCM_16")

    plot_path = save_comparison_plot(noisy_audio, enhanced_audio, stem)

    print(f"Enhanced WAV : {out_wav}")
    print(f"Plot         : {plot_path}")
    print(f"\n{GREEN}Done. Proceed to Step 8 (evaluation).{RESET}\n")
