# Voice Authentication — ECAPA-TDNN + ChromaDB

Speaker verification system based on the ECAPA-TDNN model and ChromaDB vector database. Built as part of a biometrics lab course at PWr.

## System Requirements

- Python **3.11** or **3.12**
- **ffmpeg** installed and available on `PATH` (required by tasks 6 and 7, and for loading MP3/M4A/Opus files)
- GPU optional — the model runs on CPU, but slower (~2–5× for large datasets)

### Installing ffmpeg (Windows)

```powershell
# via winget
winget install ffmpeg

# or via chocolatey
choco install ffmpeg
```

Verify the installation:

```powershell
ffmpeg -version
```

---

## Installing Dependencies

The project uses a Python virtual environment. Create it and install packages:

```powershell
# Create venv (from the lab/voice/ directory)
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> If you use a shared venv at `d:\pwr\biometria\venv`, activate it instead:
> ```powershell
> d:\pwr\biometria\venv\Scripts\Activate.ps1
> ```

---

## Project Structure

```
lab/voice/
├── config.toml              # Project configuration (paths, model, thresholds)
├── requirements.txt         # pip dependencies
│
├── src/                     # Helper modules
│   ├── config.py            # Configuration loader (cfg singleton)
│   ├── download_data.py     # Download VoxCeleb1 / LibriSpeech / VoxPopuli via HF streaming
│   │                        #   also: convert, split, download-custom subcommands
│   ├── embeddings.py        # ECAPA-TDNN embedding extraction (supports all audio formats)
│   ├── enroll.py            # Enroll / delete speakers in ChromaDB
│   ├── auth.py              # Speaker verification / identification / demo
│   ├── database.py          # ChromaDB collection operations
│   ├── preprocessing.py     # Normalization, VAD, resampling
│   └── threshold.py         # EER threshold computation, FAR/FRR metrics
│
├── notebooks/               # Lab tasks (Jupyter)
│   ├── task1_baseline.ipynb         # Task 1: baseline system, EER threshold
│   ├── task2_amplitude.ipynb        # Task 2: amplitude scaling
│   ├── task3_downsampling.ipynb     # Task 3: downsampling (naive + proper)
│   ├── task4_gaussian_noise.ipynb   # Task 4: Gaussian noise (SNR 40/20/10 dB)
│   ├── task5_background_noise.ipynb # Task 5: background noise (SNR 20/10/0 dB)
│   ├── task6_compression.ipynb      # Task 6: lossy compression (MP3/AAC/Opus)
│   └── task7_reverberation.ipynb    # Task 7: reverberation (RT60 0.2–2.0 s)
│
├── data/
│   ├── voxceleb1/           # All downloaded speaker audio
│   │                        #   VoxCeleb → id10xxx/   LibriSpeech → ls_*/   VoxPopuli → vp_*/
│   ├── custom/              # Group member recordings (<speaker_id>/*.wav)
│   ├── enrollment/          # Enrollment split — 5 files per speaker
│   └── test/                # Test split — remaining files (never seen during enrollment)
│
├── db/                      # ChromaDB persistent store (created during enrollment)
├── models/                  # ECAPA-TDNN checkpoints (downloaded automatically)
├── results/                 # JSON results and PNG plots from each notebook
└── report/
    └── raport.md            # Project report
```

---

## Running Order

### Step 1 — Download the data

The script streams audio on-demand from HuggingFace — no full archive download required.

**VoxCeleb1 + LibriSpeech (default):**

```powershell
cd d:\pwr\biometria\lab\voice
python src/download_data.py download-hf --n-speakers 140 --n-voxceleb 40 --n-librispeech 100
```

**All three sources (VoxCeleb + LibriSpeech + VoxPopuli):**

```powershell
python src/download_data.py download-hf --source all --n-speakers 180 --n-voxceleb 40 --n-librispeech 100 --n-voxpopuli 40
```

| Source | HuggingFace dataset | Language | Speaker prefix |
|---|---|---|---|
| VoxCeleb1 | `asahi417/voxceleb1-test-split` | multilingual (celebrities) | `id10xxx` |
| LibriSpeech | `openslr/librispeech_asr` | English (read speech) | `ls_` |
| VoxPopuli | `facebook/voxpopuli` | English (parliamentary) | `vp_` |

### Step 2 — Run notebooks in order

```powershell
jupyter lab
```

Run notebooks **in order** — task1 must go first (generates EER threshold and ChromaDB collection):

| Notebook | What it does | Output |
|---|---|---|
| `task1_baseline.ipynb` | Enrolls speakers, computes EER threshold | `results/task1_results.json`, `db/` |
| `task2_amplitude.ipynb` | Amplitude scaling ×0.04 / ×1 / ×25 | `results/task2_results.json` |
| `task3_downsampling.ipynb` | Naive and proper downsampling ÷2/÷5/÷10 | `results/task3_results.json` |
| `task4_gaussian_noise.ipynb` | Gaussian noise at SNR 40/20/10 dB | `results/task4_results.json` |
| `task5_background_noise.ipynb` | Background noise at SNR 20/10/0 dB | `results/task5_results.json` |
| `task6_compression.ipynb` | MP3/AAC/Opus at various bitrates | `results/task6_results.json` |
| `task7_reverberation.ipynb` | Synthetic reverb at RT60 0.2–2.0 s | `results/task7_results.json` |

---

## Adding Custom (Group Member) Recordings

### Download from YouTube

```powershell
python src/download_data.py download-custom <speaker_id> <url1> <url2> ...
```

Example:

```powershell
python src/download_data.py download-custom maklowicz https://www.youtube.com/shorts/LVAyOL9k_Ps
```

Files are saved to `data/custom/<speaker_id>/` as 16 kHz mono WAV.

### Convert existing MP3/M4A/OGG files

If you have recordings in a non-WAV format, convert them first:

```powershell
# Convert in place (keeps originals)
python src/download_data.py convert data/custom/<speaker_id>

# Convert and delete originals
python src/download_data.py convert data/custom/<speaker_id> --remove-originals

# Convert to a separate output folder
python src/download_data.py convert data/custom/<speaker_id> --out-dir data/custom/<speaker_id>_wav
```

Supported input formats: `.mp3` `.m4a` `.ogg` `.opus` `.flac`

### Split into enrollment / test

```powershell
python src/download_data.py split --speaker-root data/custom
```

Requires at least **6 files** per speaker (5 enrollment + 1 test). To lower the minimum, set `enrollment_samples_per_speaker = 3` in [config.toml](config.toml).

### Enroll into the database

```powershell
cd src

# Single speaker
python enroll.py --speaker-id maklowicz --audio-dir ..\data\enrollment\maklowicz

# All speakers in data/enrollment/ at once
python enroll.py
```

---

## Enrollment Management

All commands run from `src/`:

```powershell
cd src

# List enrolled speakers
python enroll.py --list

# Re-enroll (overwrite existing profiles)
python enroll.py --overwrite

# Enroll all except LibriSpeech speakers
python enroll.py --exclude ls_

# Remove a single speaker
python enroll.py --delete maklowicz

# Remove all speakers with a given prefix (e.g. all LibriSpeech)
python enroll.py --delete-prefix ls_
python enroll.py --delete-prefix ls_ --yes   # skip confirmation

# Remove everyone
python enroll.py --delete-all
python enroll.py --delete-all --yes          # skip confirmation
```

---

## Authentication CLI

All commands run from `src/`:

```powershell
cd src

# 1-to-1 verification: is this audio from user_id?
python auth.py verify <user_id> <audio_file>

# 1-to-N identification: who is speaking?
python auth.py identify <audio_file>

# Live demo — pretty-prints top-5 candidates + optional verification
python auth.py demo <audio_file>
python auth.py demo <audio_file> --user-id <speaker_id>
```

The `demo` command accepts any audio format (WAV, MP3, M4A, OGG, Opus) — ffmpeg handles the decoding automatically.

Example output:

```
========================================================
  VOICE AUTHENTICATION DEMO
========================================================
  File      : recording.wav
  Duration  : 12.40 s
  Threshold : 0.4031
--------------------------------------------------------

  [VERIFY]  claimed identity: maklowicz
  Score     : 0.7823  [###############-----]
                       ^threshold
  Result    : ACCEPTED

  [IDENTIFY]  top-5 candidates
  1. maklowicz           [###############-----]  0.7823  <-- match
  2. id10270             [########------------]  0.4102
  3. vp_96718            [######--------------]  0.3241
  4. filip               [####----------------]  0.2187
  5. ls_1034             [###-----------------]  0.1876
========================================================
```

---

## Configuration

[config.toml](config.toml) holds all project parameters:

```toml
[model]
source = "speechbrain/spkrec-ecapa-voxceleb"   # model from HuggingFace Hub
embedding_dim = 192

[audio]
sample_rate = 16000

[dataset]
num_speakers = 100
enrollment_samples_per_speaker = 5   # minimum files needed for split: this + 1
random_seed = 42

[auth]
threshold = 0.25   # cosine similarity threshold (overridden by EER from task1)
```

The ECAPA-TDNN model is downloaded automatically from HuggingFace on first run and cached under `models/ecapa/`.

---

## Known Issues

**ChromaDB `InternalError: Error finding id`**
> With some ChromaDB versions using the Rust backend, calling `collection.get(ids=[...])` may raise an exception. All notebooks use a safe workaround — a single bulk load via `collection.get(include=["embeddings"])` with no ID filter.

**Missing `task1_results.json`**
> Tasks 2–7 load the EER threshold from `results/task1_results.json`. If the file does not exist, run `task1_baseline.ipynb` to completion first.

**`blemondo`/`miku`/custom speaker shows 0 enrollment files**
> The split requires at least `enrollment_samples_per_speaker + 1` audio files. Lower the threshold in `config.toml` or add more recordings, then re-run `split --speaker-root data/custom`.

**Audio path not found in `auth.py demo`**
> Always pass the full or relative path from the project root, not just the filename:
> ```powershell
> python src/auth.py demo .\data\test\maklowicz\LVAyOL9k_Ps.wav --user-id maklowicz
> ```

---

## Dependencies Overview

| Package | Role |
|---|---|
| `torch`, `torchaudio` | Tensors, resampling, convolution (RIR) |
| `speechbrain` | ECAPA-TDNN model |
| `chromadb` | Vector database for speaker embeddings |
| `soundfile` | Audio I/O backend for WAV/FLAC/OGG |
| `numpy`, `scipy` | Numerical computation, EER/ROC metrics |
| `matplotlib` | Plots (score distributions, ROC, DET curves) |
| `datasets`, `huggingface_hub` | Streaming download of VoxCeleb1 / LibriSpeech / VoxPopuli |
| `imageio-ffmpeg` | Bundled ffmpeg binary for MP3/M4A/Opus decoding and compression tasks |
| `ffmpeg` (system) | Audio encoding/decoding (MP3, AAC, Opus) for task 6 |
