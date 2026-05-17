"""
Audio preprocessing utilities.

Applied before embedding extraction to improve signal quality:
  1. Amplitude normalization  – rescale to target RMS or peak level
  2. VAD (Voice Activity Detection) – strip leading/trailing silence
     Uses energy-based approach; no extra model required.
  3. Chunk splitting – split long recordings into fixed-length segments

All functions accept a waveform tensor (1, T) at SAMPLE_RATE and return
a processed tensor of the same shape (or shorter after VAD/chunking).
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path

from config import cfg

SAMPLE_RATE = cfg.audio.sample_rate

# VAD defaults
VAD_FRAME_MS    = 20      # frame length for energy computation (ms)
VAD_HOP_MS      = 10      # hop between frames (ms)
VAD_ENERGY_THR  = 0.005   # fraction of max energy below which frame is silence
VAD_PAD_MS      = 150     # silence padding kept around speech (ms)

# Normalization default
TARGET_RMS_DB = -23.0     # EBU R128 loudness target (dB)


# ---------------------------------------------------------------------------
# Amplitude normalization
# ---------------------------------------------------------------------------

def normalize_rms(waveform: torch.Tensor, target_db: float = TARGET_RMS_DB) -> torch.Tensor:
    """
    Scale waveform so its RMS power equals target_db dBFS.
    Clips at ±1.0 to avoid integer overflow.
    """
    rms = waveform.pow(2).mean().sqrt()
    if rms < 1e-9:
        return waveform  # silent signal – do not scale
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms.item()
    return (waveform * gain).clamp(-1.0, 1.0)


def normalize_peak(waveform: torch.Tensor, target_peak: float = 0.9) -> torch.Tensor:
    """Scale waveform so the peak absolute value equals target_peak."""
    peak = waveform.abs().max()
    if peak < 1e-9:
        return waveform
    return waveform * (target_peak / peak.item())


# ---------------------------------------------------------------------------
# Energy-based VAD
# ---------------------------------------------------------------------------

def apply_vad(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = VAD_FRAME_MS,
    hop_ms: int = VAD_HOP_MS,
    energy_thr: float = VAD_ENERGY_THR,
    pad_ms: int = VAD_PAD_MS,
) -> torch.Tensor:
    """
    Remove silence from the beginning and end of the waveform.

    Splits into short frames, marks frames with energy > threshold as speech,
    then returns the slice from first to last speech frame with padding.

    Returns original waveform unchanged if no speech is detected.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    hop_len   = int(sample_rate * hop_ms  / 1000)
    pad_len   = int(sample_rate * pad_ms  / 1000)

    signal = waveform.squeeze(0).numpy()
    n = len(signal)

    # Compute per-frame RMS energy
    frames = []
    pos = 0
    while pos + frame_len <= n:
        frame = signal[pos : pos + frame_len]
        frames.append(np.sqrt(np.mean(frame ** 2)))
        pos += hop_len

    if not frames:
        return waveform

    energies = np.array(frames)
    threshold = energies.max() * energy_thr
    speech_mask = energies > threshold

    if not speech_mask.any():
        return waveform  # fully silent – return as-is

    first_frame = int(np.argmax(speech_mask))
    last_frame  = int(len(speech_mask) - 1 - np.argmax(speech_mask[::-1]))

    start = max(0, first_frame * hop_len - pad_len)
    end   = min(n, last_frame  * hop_len + frame_len + pad_len)

    trimmed = signal[start:end]
    return torch.tensor(trimmed, dtype=waveform.dtype).unsqueeze(0)


# ---------------------------------------------------------------------------
# Chunk splitting
# ---------------------------------------------------------------------------

def split_into_chunks(
    waveform: torch.Tensor,
    chunk_s: float = 3.0,
    min_chunk_s: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
    overlap_s: float = 0.5,
) -> list[torch.Tensor]:
    """
    Split a long waveform into overlapping fixed-length chunks.

    Args:
        chunk_s:     target chunk length in seconds
        min_chunk_s: drop final chunk if shorter than this
        overlap_s:   overlap between consecutive chunks in seconds

    Returns list of (1, T) tensors.
    """
    chunk_len   = int(sample_rate * chunk_s)
    hop_len     = int(sample_rate * (chunk_s - overlap_s))
    min_len     = int(sample_rate * min_chunk_s)

    signal = waveform.squeeze(0)
    n = len(signal)

    if n <= chunk_len:
        return [waveform]

    chunks = []
    pos = 0
    while pos + min_len <= n:
        end = min(pos + chunk_len, n)
        chunk = signal[pos:end].unsqueeze(0)
        chunks.append(chunk)
        pos += hop_len

    return chunks


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    audio_path: str | Path,
    apply_normalization: bool = True,
    apply_vad_trim: bool = True,
    target_db: float = TARGET_RMS_DB,
) -> torch.Tensor:
    """
    Load an audio file and apply the standard preprocessing pipeline:
      load → mono → resample → normalize → VAD trim

    Returns a (1, T) float32 tensor ready for embedding extraction.
    """
    waveform, sr = torchaudio.load(str(audio_path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    if apply_normalization:
        waveform = normalize_rms(waveform, target_db)

    if apply_vad_trim:
        waveform = apply_vad(waveform)

    return waveform


def preprocess_and_save(
    audio_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> None:
    """Preprocess audio and save the result as a WAV file."""
    waveform = preprocess(audio_path, **kwargs)
    torchaudio.save(str(output_path), waveform, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# CLI – batch preprocess a directory
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Batch preprocess audio files")
    parser.add_argument("input_dir",  type=Path, help="Directory with WAV files")
    parser.add_argument("output_dir", type=Path, help="Where to save processed files")
    parser.add_argument("--no-vad",   action="store_true", help="Skip VAD trimming")
    parser.add_argument("--no-norm",  action="store_true", help="Skip amplitude normalization")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"ERROR: {args.input_dir} does not exist")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.rglob("*.wav"))
    print(f"Processing {len(files)} file(s)...")

    for f in files:
        rel = f.relative_to(args.input_dir)
        out = args.output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        preprocess_and_save(
            f, out,
            apply_normalization=not args.no_norm,
            apply_vad_trim=not args.no_vad,
        )
        print(f"  [ok] {rel}")

    print("Done.")
