"""
Step 2 — Dataset Organizer
Scans data/clean/ and data/noise/, converts any non-WAV files to WAV,
resamples everything to 16kHz mono, and prints a summary report.
"""

import os
import sys
import shutil
import soundfile as sf
import librosa
import numpy as np

TARGET_SR = 16000

CLEAN_DIR = "data/clean"
NOISE_DIRS = {
    "engine": "data/noise/engine",
    "cockpit": "data/noise/cockpit",
    "radio": "data/noise/radio",
}

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def resample_and_save(src_path, dst_path):
    """Load audio at any format/rate, resample to 16kHz mono, save as WAV."""
    audio, sr = librosa.load(src_path, sr=TARGET_SR, mono=True)
    sf.write(dst_path, audio, TARGET_SR, subtype="PCM_16")


def process_directory(directory, label):
    """Convert and resample all audio files in a directory in-place."""
    all_files = [
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not all_files:
        print(f"  {YELLOW}[WARN]{RESET} No audio files found in {directory}")
        return 0

    converted = 0
    already_wav = 0
    errors = 0

    for filename in all_files:
        src = os.path.join(directory, filename)
        stem, ext = os.path.splitext(filename)
        dst = os.path.join(directory, stem + ".wav")

        try:
            if ext.lower() != ".wav":
                resample_and_save(src, dst)
                os.remove(src)
                converted += 1
                print(f"  {GREEN}[CONV]{RESET} {filename} → {stem}.wav")
            else:
                # Already WAV — check sample rate and resample if needed
                info = sf.info(src)
                if info.samplerate != TARGET_SR or info.channels != 1:
                    tmp = src + ".tmp.wav"
                    resample_and_save(src, tmp)
                    os.replace(tmp, src)
                    converted += 1
                    print(f"  {GREEN}[RESAMP]{RESET} {filename} ({info.samplerate}Hz → {TARGET_SR}Hz)")
                else:
                    already_wav += 1
        except Exception as e:
            print(f"  {RED}[ERR]{RESET} {filename} — {e}")
            errors += 1

    total = len(all_files) - errors
    print(f"  → {label}: {total} files ready ({converted} converted/resampled, {already_wav} already correct, {errors} errors)")
    return total


def count_wav_files(directory):
    return len([f for f in os.listdir(directory) if f.lower().endswith(".wav")])


print(f"\n{BOLD}=== Aviation Speech Enhancement — Step 2: Dataset Organizer ==={RESET}\n")

# --- Clean speech ---
print(f"{BOLD}Clean speech  ({CLEAN_DIR}){RESET}")
clean_count = process_directory(CLEAN_DIR, "clean")

# --- Noise by category ---
print(f"\n{BOLD}Noise samples{RESET}")
noise_counts = {}
for category, path in NOISE_DIRS.items():
    noise_counts[category] = process_directory(path, category)

# --- Summary ---
total_noise = sum(noise_counts.values())
print(f"\n{'=' * 55}")
print(f"{BOLD}Summary{RESET}")
print(f"  Clean files   : {count_wav_files(CLEAN_DIR)}")
for cat, path in NOISE_DIRS.items():
    print(f"  Noise/{cat:<8}: {count_wav_files(path)}")
print(f"  Total noise   : {total_noise}")
print(f"{'=' * 55}")

if clean_count == 0 or total_noise == 0:
    print(f"\n{RED}Missing data — see instructions above for what to download.{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}Dataset ready. Proceed to Step 3.{RESET}\n")
