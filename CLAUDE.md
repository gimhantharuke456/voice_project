# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aviation Speech Enhancement System — a Convolutional Denoising Autoencoder (CDAE) that removes cockpit/engine/radio noise from pilot-ATC audio and exposes the model as a Flask API.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torchaudio`, `librosa`, `soundfile`, `numpy`, `scipy`, `matplotlib`, `pesq`, `pystoi`, `flask`

## Running the Pipeline (Build Order)

The project is built in sequential steps — each step's output feeds the next:

```bash
python step1_setup_check.py          # verify env & create folder structure
python step2_dataset_organizer.py    # normalize raw audio to 16kHz WAV
python step3_noise_simulator.py      # generate noisy-clean pairs at -5/0/5/10 dB SNR
python step4_preprocessor.py         # audio → STFT spectrogram tensors
python step5_model.py                # verify model architecture
python step6_trainer.py              # train CDAE, saves models/cdae_best.pth
python step7_reconstructor.py --input data/generated/noisy/sample.wav
python step8_evaluator.py --noisy <noisy.wav> --enhanced <enhanced.wav> --clean <clean.wav>
python step9_gui.py                  # launch Tkinter desktop GUI
python step10_handover_check.py      # pre-delivery checklist
```

## Running the Flask API

```bash
python api/app.py                    # starts on http://localhost:5000
```

Endpoints:
- `POST /enhance` — multipart WAV upload → returns enhanced WAV
- `POST /evaluate` — noisy + clean WAV → returns JSON with SNR/PESQ/STOI scores

## Architecture

### Audio Processing Pipeline

All audio is standardized to **16kHz mono WAV**. The spectrogram pipeline uses:
- FFT size (`n_fft`): 512
- Hop length: 128
- Window: Hann
- Phase is saved separately from magnitude for ISTFT reconstruction (Griffin-Lim fallback also supported)

### CDAE Model (`step5_model.py`)

```
Input (noisy spectrogram)
↓
[Conv2D → ReLU → MaxPool] × 3   ← Encoder
↓
[Bottleneck Conv2D]
↓
[ConvTranspose2D → ReLU] × 3    ← Decoder
↓
Output (clean spectrogram)
```

Loss: MSE. Optimizer: Adam. Best weights saved to `models/cdae_best.pth`.

### Flask API Structure (`api/`)

- `app.py` — creates Flask app, registers blueprints, configures uploads
- `routes/enhance.py` — POST /enhance route
- `routes/evaluate.py` — POST /evaluate route
- `services/preprocess.py` — `load_audio()`, `to_spectrogram()`, `normalize()`
- `services/denoise.py` — `load_model()` (called once at startup), `enhance()`
- `services/metrics.py` — `compute_snr()`, `compute_pesq()`, `compute_stoi()`
- `utils/file_handler.py` — upload save/cleanup/validation helpers
- `config.py` — all paths, audio params, and model hyperparameters in one place

### Data Layout

```
data/
  clean/              # LibriSpeech / TIMIT source files
  noise/
    engine/
    cockpit/
    radio/
  generated/
    noisy/            # naming: utterance001_engine_0dB.wav
    clean/            # matching clean reference per noisy file
models/
  cdae_best.pth
outputs/
  enhanced/           # reconstructed WAV files
  plots/              # waveform & spectrogram PNGs
```

## Key Constraints

- All audio must be resampled to **16kHz mono** before any processing step
- Noisy-clean file pairs share the same name prefix (used for matching during evaluation)
- Training is GPU-intensive — Step 6 is designed to run on Google Colab if no local GPU
- The model is loaded once at API startup in `services/denoise.py`, not per-request
- `api/config.py` is the single source of truth for all paths and audio parameters
