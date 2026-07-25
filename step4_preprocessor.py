"""
Step 4 — Audio Preprocessor
Converts noisy/clean WAV pairs → STFT magnitude spectrograms saved as .npy.
Phase of the noisy file is saved separately for reconstruction in Step 7.
Also defines the SpectrogramDataset class used by the trainer in Step 6.
"""

import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

# ── Audio / STFT parameters (single source of truth) ──────────────────────────
SR       = 16000
N_FFT    = 512
HOP_LEN  = 128
WIN_LEN  = N_FFT

# ── Paths ──────────────────────────────────────────────────────────────────────
NOISY_DIR  = "data/generated/noisy"
CLEAN_DIR  = "data/generated/clean"
SPEC_DIR   = "data/spectrograms"
PHASE_DIR  = "data/phases"
PLOT_PATH  = "outputs/plots/sample_spectrogram.png"

GREEN = "\033[92m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"


# ── Core functions ─────────────────────────────────────────────────────────────

def to_spectrogram(waveform):
    """Waveform → (magnitude, phase) via STFT."""
    stft   = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN)
    mag    = np.abs(stft).astype(np.float32)
    phase  = np.angle(stft).astype(np.float32)
    return mag, phase


def normalize(mag):
    """Scale magnitude to [0, 1] using log compression then min-max."""
    log_mag = np.log1p(mag)
    mn, mx  = log_mag.min(), log_mag.max()
    if mx - mn < 1e-8:
        return np.zeros_like(log_mag)
    return ((log_mag - mn) / (mx - mn)).astype(np.float32)


def load_pair(stem):
    """Load a preprocessed (noisy_spec, clean_spec) pair by stem name."""
    noisy = np.load(os.path.join(SPEC_DIR, "noisy", stem + ".npy"))
    clean = np.load(os.path.join(SPEC_DIR, "clean", stem + ".npy"))
    return noisy, clean


# ── Dataset class (used by Step 6 trainer) ────────────────────────────────────

class SpectrogramDataset(Dataset):
    def __init__(self, split="train", val_ratio=0.2):
        noisy_dir = os.path.join(SPEC_DIR, "noisy")
        stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(noisy_dir) if f.endswith(".npy")
        ])
        split_idx = int(len(stems) * (1 - val_ratio))
        self.stems = stems[:split_idx] if split == "train" else stems[split_idx:]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        noisy, clean = load_pair(self.stems[idx])
        # Add channel dim: (1, freq, time)
        return (
            torch.from_numpy(noisy).unsqueeze(0),
            torch.from_numpy(clean).unsqueeze(0),
        )


# ── Preprocessing run ──────────────────────────────────────────────────────────

def preprocess_all():
    for sub in ["noisy", "clean"]:
        os.makedirs(os.path.join(SPEC_DIR, sub), exist_ok=True)
    os.makedirs(PHASE_DIR, exist_ok=True)

    noisy_files = sorted([f for f in os.listdir(NOISY_DIR) if f.endswith(".wav")])
    if not noisy_files:
        print(f"{RED}No WAV files in {NOISY_DIR}. Run Step 3 first.{RESET}")
        sys.exit(1)

    print(f"Files to process : {len(noisy_files)}")
    print(f"STFT params      : n_fft={N_FFT}, hop={HOP_LEN}, sr={SR}\n")

    done = errors = skipped = 0

    for i, fname in enumerate(noisy_files):
        stem = os.path.splitext(fname)[0]
        noisy_out = os.path.join(SPEC_DIR, "noisy", stem + ".npy")
        clean_out = os.path.join(SPEC_DIR, "clean", stem + ".npy")

        if os.path.exists(noisy_out) and os.path.exists(clean_out):
            skipped += 1
            continue

        noisy_wav = os.path.join(NOISY_DIR, fname)
        clean_wav = os.path.join(CLEAN_DIR, fname)

        if not os.path.exists(clean_wav):
            print(f"  {RED}[MISS]{RESET} no matching clean for {fname}")
            errors += 1
            continue

        try:
            noisy_audio, _ = librosa.load(noisy_wav, sr=SR, mono=True)
            clean_audio, _ = librosa.load(clean_wav, sr=SR, mono=True)

            noisy_mag, noisy_phase = to_spectrogram(noisy_audio)
            clean_mag, _           = to_spectrogram(clean_audio)

            np.save(noisy_out,                             normalize(noisy_mag))
            np.save(clean_out,                             normalize(clean_mag))
            np.save(os.path.join(PHASE_DIR, stem + ".npy"), noisy_phase)

            done += 1
        except Exception as e:
            print(f"  {RED}[ERR]{RESET} {fname} — {e}")
            errors += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(noisy_files):
            print(f"  [{i+1}/{len(noisy_files)}] done={done}  skipped={skipped}  errors={errors}")

    return done, skipped, errors


def save_sample_plot():
    noisy_dir = os.path.join(SPEC_DIR, "noisy")
    files = [f for f in os.listdir(noisy_dir) if f.endswith(".npy")]
    if not files:
        return

    stem  = os.path.splitext(files[0])[0]
    noisy, clean = load_pair(stem)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, spec, title in zip(axes, [noisy, clean], ["Noisy spectrogram", "Clean spectrogram"]):
        img = librosa.display.specshow(spec, sr=SR, hop_length=HOP_LEN,
                                       x_axis="time", y_axis="linear", ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+.2f")

    fig.suptitle(stem, fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"\n  Sample plot saved → {PLOT_PATH}")


if __name__ == "__main__":
    print(f"\n{BOLD}=== Aviation Speech Enhancement — Step 4: Preprocessor ==={RESET}\n")

    done, skipped, errors = preprocess_all()

    print(f"\n{'='*55}")
    print(f"{BOLD}Summary{RESET}")
    print(f"  Processed : {done}")
    print(f"  Skipped   : {skipped}  (already existed)")
    print(f"  Errors    : {errors}")
    print(f"{'='*55}")

    save_sample_plot()

    # Quick dataset sanity check
    print(f"\n{BOLD}Dataset check{RESET}")
    for split in ["train", "val"]:
        ds = SpectrogramDataset(split=split)
        if len(ds) > 0:
            n, c = ds[0]
            print(f"  {split:5s}: {len(ds):5d} samples  |  shape noisy={tuple(n.shape)}  clean={tuple(c.shape)}")
        else:
            print(f"  {split:5s}: 0 samples")

    if errors == 0:
        print(f"\n{GREEN}All done. Proceed to Step 5.{RESET}\n")
    else:
        print(f"\n{RED}{errors} errors — check output above.{RESET}\n")
