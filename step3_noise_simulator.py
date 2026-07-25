"""
Step 3 — Noise Simulator
Mixes clean speech with noise at 4 SNR levels (-5, 0, 5, 10 dB).
Saves paired noisy/clean WAVs to data/generated/.
"""

import os
import sys
import random
import numpy as np
import soundfile as sf
import librosa

CLEAN_DIR = "data/generated/clean"
NOISY_DIR = "data/generated/noisy"
CLEAN_SRC = "data/clean"
NOISE_DIRS = {
    "engine":  "data/noise/engine",
    "cockpit": "data/noise/cockpit",
    "radio":   "data/noise/radio",
}
SNR_LEVELS  = [-5, 0, 5, 10]
TARGET_SR   = 16000
SEGMENT_LEN = TARGET_SR * 3  # 3-second clips
MAX_CLEAN   = 500             # cap to keep generated data under ~1.1 GB

GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def mix_at_snr(speech, noise, snr_db):
    """Scale noise so the mixture hits the target SNR, then add."""
    speech_power = np.mean(speech ** 2) + 1e-9
    noise_power  = np.mean(noise ** 2)  + 1e-9
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return np.clip(speech + scale * noise, -1.0, 1.0)


def load_segment(path, length):
    """Load a fixed-length segment from an audio file, looping if needed."""
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    if len(audio) == 0:
        raise ValueError("Empty audio file")
    # Loop until long enough, then trim
    reps = (length // len(audio)) + 2
    audio = np.tile(audio, reps)[:length]
    return audio


def collect_wav(directory):
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".wav")
    ])


print(f"\n{BOLD}=== Aviation Speech Enhancement — Step 3: Noise Simulator ==={RESET}\n")

random.seed(42)
clean_files = random.sample(collect_wav(CLEAN_SRC), min(MAX_CLEAN, len(collect_wav(CLEAN_SRC))))
noise_files  = {cat: collect_wav(path) for cat, path in NOISE_DIRS.items()}

if not clean_files:
    print(f"{RED}No WAV files in {CLEAN_SRC}. Run Step 2 first.{RESET}")
    sys.exit(1)

for cat, files in noise_files.items():
    if not files:
        print(f"{RED}No WAV files in data/noise/{cat}. Add noise samples first.{RESET}")
        sys.exit(1)

print(f"Clean files : {len(clean_files)}")
for cat, files in noise_files.items():
    print(f"Noise/{cat:<8}: {len(files)}")
print(f"SNR levels  : {SNR_LEVELS} dB")
print(f"Combinations: {len(clean_files)} × {len(NOISE_DIRS)} categories × {len(SNR_LEVELS)} SNR levels = {len(clean_files)*len(NOISE_DIRS)*len(SNR_LEVELS)} pairs\n")

pairs_written = 0
errors = 0

for idx, clean_path in enumerate(clean_files):
    stem = os.path.splitext(os.path.basename(clean_path))[0]

    try:
        speech = load_segment(clean_path, SEGMENT_LEN)
    except Exception as e:
        print(f"  {RED}[ERR]{RESET} {stem} — {e}")
        errors += 1
        continue

    for category, noise_list in noise_files.items():
        noise_path = random.choice(noise_list)

        try:
            noise = load_segment(noise_path, SEGMENT_LEN)
        except Exception as e:
            print(f"  {RED}[ERR]{RESET} noise {category} — {e}")
            errors += 1
            continue

        for snr in SNR_LEVELS:
            snr_tag = f"{snr:+d}dB".replace("+", "")
            out_name = f"{stem}_{category}_{snr_tag}.wav"

            noisy  = mix_at_snr(speech, noise, snr)
            noisy_path = os.path.join(NOISY_DIR, out_name)
            clean_out  = os.path.join(CLEAN_DIR,  out_name)

            sf.write(noisy_path, noisy,  TARGET_SR, subtype="PCM_16")
            sf.write(clean_out,  speech, TARGET_SR, subtype="PCM_16")
            pairs_written += 1

    if (idx + 1) % 100 == 0 or (idx + 1) == len(clean_files):
        print(f"  Processed {idx + 1}/{len(clean_files)} clean files — {pairs_written} pairs so far...")

print(f"\n{'=' * 55}")
print(f"{BOLD}Summary{RESET}")
print(f"  Pairs written : {pairs_written}")
print(f"  Errors        : {errors}")
print(f"{'=' * 55}")

if pairs_written == 0:
    print(f"\n{RED}No pairs generated. Check errors above.{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}Done. Proceed to Step 4.{RESET}\n")
