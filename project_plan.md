# Aviation Speech Enhancement System
### Deep Denoising Autoencoder for Pilot-ATC Communication

> **Client:** Ngwmal Seneviratha (C/ENG/23/6518/AE)  
> **Degree:** BSc Aeronautical Engineering — General Sir John Kotelawala Defense University  
> **Freelancer Build:** 10-step Python project ending with a Flask API backend

---

## What We're Building

A Python-based AI software system that takes noisy aviation audio (engine noise, cockpit noise, radio interference) and outputs clean, enhanced speech using a **Convolutional Denoising Autoencoder (CDAE)**. The final product is a **Flask API** that the client can run locally or deploy, accepting a noisy audio file and returning an enhanced version along with evaluation scores.

---

## Project Folder Structure

```
aviation-speech-enhancement/
│
├── data/
│   ├── clean/                  # Raw clean speech (LibriSpeech, TIMIT)
│   ├── noise/                  # Noise samples (engine, cockpit, radio)
│   └── generated/              # Mixed noisy-clean pairs
│       ├── noisy/
│       └── clean/
│
├── models/
│   └── cdae_best.pth           # Saved trained model weights
│
├── outputs/
│   ├── enhanced/               # Enhanced WAV output files
│   └── plots/                  # Waveform & spectrogram images
│
├── step1_setup_check.py
├── step2_dataset_organizer.py
├── step3_noise_simulator.py
├── step4_preprocessor.py
├── step5_model.py
├── step6_trainer.py
├── step7_reconstructor.py
├── step8_evaluator.py
├── step9_gui.py
├── step10_handover_check.py
│
├── api/
│   ├── app.py                  # Flask entry point
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── enhance.py          # POST /enhance
│   │   └── evaluate.py         # POST /evaluate
│   ├── services/
│   │   ├── __init__.py
│   │   ├── preprocess.py       # Audio preprocessing logic
│   │   ├── denoise.py          # Load model & run inference
│   │   └── metrics.py          # SNR, PESQ, STOI calculation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── file_handler.py     # Upload/save/cleanup helpers
│   └── config.py               # Paths, model settings, constants
│
├── requirements.txt
└── README.md
```

---

## Step-by-Step Build Guide

---

### Step 1 — `step1_setup_check.py` · Environment Verification

**Purpose:** Confirm the development environment is correctly configured before any real work begins.

**What this file does:**
- Checks that all required Python packages are installed (torch, librosa, soundfile, numpy, pesq, pystoi, matplotlib, flask)
- Prints the versions of key libraries to console
- Verifies that the folder structure (`data/`, `models/`, `outputs/`) exists and creates them if missing
- Optionally checks if a CUDA-capable GPU is available for faster training

**How to run:**
```bash
python step1_setup_check.py
```

**Expected output:** A green checklist in the terminal confirming all dependencies and folders are ready. If anything is missing, it tells you exactly what to install.

**Libraries needed:** `torch`, `librosa`, `soundfile`, `numpy`, `pesq`, `pystoi`, `matplotlib`, `flask`

---

### Step 2 — `step2_dataset_organizer.py` · Dataset Collection & Organization

**Purpose:** Organize the raw downloaded audio into a clean, consistent structure the rest of the pipeline can rely on.

**What this file does:**
- Scans the `data/clean/` folder for audio files from LibriSpeech or TIMIT
- Scans the `data/noise/` folder and categorizes noise files into three types: engine, cockpit, radio
- Converts any non-WAV files to WAV format
- Resamples all audio to **16kHz mono** (the standard for this project)
- Prints a summary report: how many clean files, how many noise files per category

**Datasets to download manually before running:**
- Clean speech: [LibriSpeech](https://www.openslr.org/12) (use `train-clean-100`) or [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)
- Noise: [freesound.org](https://freesound.org) — search "aircraft engine", "cockpit ambient", "radio static"

**How to run:**
```bash
python step2_dataset_organizer.py
```

**Expected output:** All audio files normalized to 16kHz WAV and organized inside `data/clean/` and `data/noise/engine/`, `data/noise/cockpit/`, `data/noise/radio/`.

---

### Step 3 — `step3_noise_simulator.py` · Noise Simulation & Dataset Generation

**Purpose:** Create the paired noisy-clean training dataset by mathematically mixing clean speech with noise at controlled SNR levels.

**What this file does:**
- For each clean speech file, randomly picks a noise sample from each category
- Mixes them using the formula: `y(t) = x(t) + n(t)` scaled to target SNR
- Generates mixtures at **4 SNR levels: -5 dB, 0 dB, 5 dB, 10 dB**
- Saves paired files into `data/generated/noisy/` and `data/generated/clean/`
- Naming convention: `utterance001_engine_0dB.wav` (so pairs are easy to match)
- Prints total number of generated pairs

**Key concept — SNR formula:**
```
SNR (dB) = 10 × log10(power of signal / power of noise)
```
The script scales noise amplitude to hit the target SNR before mixing.

**How to run:**
```bash
python step3_noise_simulator.py
```

**Expected output:** Hundreds of paired WAV files in `data/generated/`. Each noisy file has a matching clean file with the same name prefix.

---

### Step 4 — `step4_preprocessor.py` · Audio Preprocessing Pipeline

**Purpose:** Convert raw audio files into spectrogram tensors that the CDAE model can process.

**What this file does:**
- Loads WAV files from `data/generated/`
- Applies **Short-Time Fourier Transform (STFT)** to convert waveform → spectrogram
- Takes the magnitude of the spectrogram (discards phase, saves it separately for reconstruction later)
- Normalizes spectrogram values to a consistent range (0 to 1)
- Saves processed spectrogram pairs as `.npy` files or packages them into a PyTorch Dataset class
- Also includes a `load_pair()` function used during training

**Key parameters (set in `api/config.py`):**
- Sample rate: `16000 Hz`
- FFT size (n_fft): `512`
- Hop length: `128`
- Window: `Hann window`

**How to run:**
```bash
python step4_preprocessor.py
```

**Expected output:** Preprocessed spectrogram pairs ready for the model. A sample spectrogram plot is saved to `outputs/plots/sample_spectrogram.png` for visual verification.

---

### Step 5 — `step5_model.py` · CDAE Model Architecture

**Purpose:** Define the neural network — the brain of the entire system.

**What this file does:**
- Defines the `CDAE` class using PyTorch (`nn.Module`)
- **Encoder:** Stack of Conv2D layers that compress the noisy spectrogram and extract features
- **Bottleneck:** The deepest compressed representation of the audio features
- **Decoder:** Stack of ConvTranspose2D (upsampling) layers that reconstruct the clean spectrogram
- Includes a `forward()` method: noisy spectrogram in → clean spectrogram out
- Prints a model summary showing layer sizes and total parameter count

**Architecture overview:**
```
Input (noisy spectrogram)
    ↓
[Conv2D → ReLU → MaxPool] × 3     ← Encoder
    ↓
[Bottleneck Conv2D]
    ↓
[ConvTranspose2D → ReLU] × 3      ← Decoder
    ↓
Output (clean spectrogram)
```

**How to run:**
```bash
python step5_model.py
```

**Expected output:** Model architecture printed to console, confirming all layers are correctly connected and output shape matches input shape.

---

### Step 6 — `step6_trainer.py` · Model Training & Validation

**Purpose:** Train the CDAE model on the generated dataset and save the best weights.

**What this file does:**
- Loads paired spectrogram data using the Dataset class from Step 4
- Splits data into **80% training / 20% validation**
- Instantiates the CDAE model from Step 5
- Sets up the **MSE loss function** and **Adam optimizer**
- Runs training for N epochs, printing loss every epoch
- Evaluates on the validation set after each epoch
- Saves the best model weights to `models/cdae_best.pth` (based on lowest validation loss)
- Saves a training loss curve plot to `outputs/plots/loss_curve.png`

**Recommended training setup:**
- Use **Google Colab** with GPU runtime if your local machine is CPU-only
- Upload your `data/generated/` folder and run this script there
- Download `models/cdae_best.pth` when done

**How to run:**
```bash
python step6_trainer.py
```

**Expected output:** Per-epoch loss printed to console, `cdae_best.pth` saved in `models/`, and a loss curve image saved in `outputs/plots/`.

---

### Step 7 — `step7_reconstructor.py` · Speech Reconstruction

**Purpose:** Take the model's output spectrogram and convert it back into a listenable audio file.

**What this file does:**
- Loads a noisy WAV file (test input)
- Runs it through the full pipeline: preprocess → model inference → output spectrogram
- Applies **Inverse STFT (ISTFT)** using the saved phase from Step 4 to convert spectrogram back to waveform
- Saves the enhanced WAV file to `outputs/enhanced/`
- Also saves a side-by-side waveform plot (noisy vs enhanced) to `outputs/plots/`

**Phase reconstruction note:**
Since we discarded phase during preprocessing, we reconstruct it using the **Griffin-Lim algorithm** or by reusing the noisy file's original phase — both are valid approaches and this file supports both.

**How to run:**
```bash
python step7_reconstructor.py --input data/generated/noisy/sample.wav
```

**Expected output:** A clean-sounding enhanced WAV in `outputs/enhanced/` and a comparison waveform plot saved to `outputs/plots/`.

---

### Step 8 — `step8_evaluator.py` · Performance Evaluation

**Purpose:** Measure how well the model performed using three standard speech quality metrics.

**What this file does:**
- Takes pairs of (noisy, enhanced, clean) WAV files
- Calculates three metrics for both noisy and enhanced vs the clean reference:
  - **SNR** (Signal-to-Noise Ratio) — how much noise is left
  - **PESQ** (Perceptual Evaluation of Speech Quality) — perceptual quality score (–0.5 to 4.5)
  - **STOI** (Short-Time Objective Intelligibility) — intelligibility score (0 to 1)
- Prints a comparison table: Noisy Score vs Enhanced Score for all three metrics
- Saves before/after spectrogram images to `outputs/plots/` for the client's academic report
- Optionally runs batch evaluation across the entire test set and saves a CSV summary

**How to run:**
```bash
python step8_evaluator.py --noisy data/generated/noisy/sample.wav \
                           --enhanced outputs/enhanced/sample.wav \
                           --clean data/generated/clean/sample.wav
```

**Expected output:** A printed metrics table and saved spectrogram comparison images ready to paste into the client's university report.

---

### Step 9 — `step9_gui.py` · Desktop GUI Application

**Purpose:** Wrap the whole system into a simple desktop app the client can demo during their university presentation.

**What this file does:**
- Builds a **Tkinter** desktop window with:
  - A **Load Audio** button — opens a file picker to select a noisy WAV
  - A **Enhance Speech** button — runs the full pipeline (preprocess → model → reconstruct)
  - A **Play Original** and **Play Enhanced** button — plays audio back
  - A metrics panel showing SNR, PESQ, STOI scores after enhancement
  - A spectrogram display area showing before/after visuals
- Loads the trained model from `models/cdae_best.pth` on startup
- Handles errors gracefully (e.g., wrong file format, model not found)

**How to run:**
```bash
python step9_gui.py
```

**Expected output:** A working desktop window. Load a WAV, click enhance, hear the difference, see the scores.

---

### Step 10 — `step10_handover_check.py` · Final Handover Verification

**Purpose:** A pre-delivery checklist script that confirms everything is in place before handing over to the client.

**What this file does:**
- Checks that `models/cdae_best.pth` exists and is loadable
- Runs a quick inference on a sample file end-to-end and confirms output is produced
- Verifies all required folders exist with expected contents
- Checks all required packages are installed at correct versions
- Prints a final **PASS / FAIL** checklist for each item
- Generates a `HANDOVER_REPORT.txt` summarizing: model details, evaluation scores achieved, known limitations, and how to run the software

**How to run:**
```bash
python step10_handover_check.py
```

**Expected output:** A printed checklist, all items marked PASS, and a `HANDOVER_REPORT.txt` ready to deliver to the client alongside the project files.

---

## Flask API Backend

The Flask API is the final integration layer — it exposes the denoising system as HTTP endpoints so the client (or future developers) can interact with it programmatically.

---

### `api/config.py` · Configuration

Stores all constants in one place:
- Paths to model weights, output folders, upload folder
- Audio parameters: sample rate, n_fft, hop length
- Model hyperparameters

---

### `api/app.py` · Flask Entry Point

- Creates the Flask app
- Registers the `enhance` and `evaluate` blueprints
- Configures upload folder and max file size
- Runs the development server on `http://localhost:5000`

**To start the API:**
```bash
python api/app.py
```

---

### `api/routes/enhance.py` · POST `/enhance`

**Accepts:** A multipart form upload with a noisy WAV file  
**Does:** Preprocesses → runs model inference → reconstructs audio  
**Returns:** The enhanced WAV file as a downloadable response

```
POST http://localhost:5000/enhance
Content-Type: multipart/form-data
Body: file=<noisy_audio.wav>

Response: enhanced_audio.wav (audio/wav)
```

---

### `api/routes/evaluate.py` · POST `/evaluate`

**Accepts:** Two WAV files — noisy and a clean reference  
**Does:** Calculates SNR, PESQ, and STOI scores  
**Returns:** A JSON response with all three scores

```
POST http://localhost:5000/evaluate
Content-Type: multipart/form-data
Body: noisy=<noisy.wav>, clean=<clean_reference.wav>

Response:
{
  "snr_noisy": 2.3,
  "snr_enhanced": 14.7,
  "pesq_noisy": 1.2,
  "pesq_enhanced": 3.4,
  "stoi_noisy": 0.61,
  "stoi_enhanced": 0.88
}
```

---

### `api/services/preprocess.py` · Preprocessing Service

Reusable functions called by the routes:
- `load_audio(path)` — load WAV at 16kHz
- `to_spectrogram(waveform)` — apply STFT, return magnitude + phase
- `normalize(spectrogram)` — scale to model input range

---

### `api/services/denoise.py` · Denoising Service

- `load_model()` — loads `cdae_best.pth` into memory once at startup
- `enhance(spectrogram)` — runs model inference, returns clean spectrogram

---

### `api/services/metrics.py` · Metrics Service

- `compute_snr(clean, enhanced)` — returns dB value
- `compute_pesq(clean, enhanced, sr)` — returns PESQ score
- `compute_stoi(clean, enhanced, sr)` — returns STOI score

---

### `api/utils/file_handler.py` · File Utilities

- `save_upload(file)` — saves uploaded file to temp folder, returns path
- `cleanup(path)` — deletes temp file after processing
- `validate_wav(path)` — checks file is a valid WAV before processing

---

## Requirements

Save this as `requirements.txt`:

```
torch
torchaudio
librosa
soundfile
numpy
scipy
matplotlib
pesq
pystoi
flask
```

Install everything with:
```bash
pip install -r requirements.txt
```

---

## Build Order Summary

| Step | File | Purpose |
|------|------|---------|
| 1 | `step1_setup_check.py` | Verify environment & folders |
| 2 | `step2_dataset_organizer.py` | Organize & convert raw audio |
| 3 | `step3_noise_simulator.py` | Generate noisy-clean pairs |
| 4 | `step4_preprocessor.py` | Audio → spectrogram pipeline |
| 5 | `step5_model.py` | Define CDAE architecture |
| 6 | `step6_trainer.py` | Train & save model |
| 7 | `step7_reconstructor.py` | Spectrogram → audio output |
| 8 | `step8_evaluator.py` | SNR / PESQ / STOI metrics |
| 9 | `step9_gui.py` | Desktop GUI for demo |
| 10 | `step10_handover_check.py` | Final delivery check |
| — | `api/` | Flask API wrapping everything |

---

*Built for academic and research purposes — General Sir John Kotelawala Defense University, 2026*