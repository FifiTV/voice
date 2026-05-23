"""
ECAPA-TDNN speaker embedding extraction using SpeechBrain pretrained model.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
# from speechbrain.pretrained import SpeakerRecognition
from speechbrain.inference.speaker import SpeakerRecognition

from config import cfg

MODEL_SOURCE  = cfg.model.source
SAMPLE_RATE   = cfg.audio.sample_rate
EMBEDDING_DIM = cfg.model.embedding_dim


def load_model(save_dir: str | Path = cfg.model.save_dir) -> SpeakerRecognition:
    """Load pretrained ECAPA-TDNN model from SpeechBrain hub."""
    model = SpeakerRecognition.from_hparams(
        source=MODEL_SOURCE,
        savedir=save_dir,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    return model


def load_audio(audio_path: str | Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Load any audio format to mono 16kHz tensor.

    Tries torchaudio first (WAV/FLAC/OGG native). Falls back to ffmpeg decoding
    for formats soundfile cannot read (MP3, M4A, Opus, AAC, etc.).
    """
    import tempfile, subprocess
    audio_path = Path(audio_path).resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            result = subprocess.run(
                [ffmpeg, "-y", "-i", str(audio_path), "-c:a", "pcm_s16le", str(tmp_path)],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed to decode {audio_path.name}:\n"
                    + result.stderr.decode(errors="replace")
                )
            waveform, sr = torchaudio.load(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform


def get_embedding(
    model: SpeakerRecognition,
    audio_path: str | Path,
) -> np.ndarray:
    """
    Extract L2-normalized speaker embedding for a single audio file.
    Returns float32 numpy array of shape (EMBEDDING_DIM,).
    """
    waveform = load_audio(audio_path)

    with torch.no_grad():
        # encode_batch expects (batch, time) — squeeze channel dim
        embedding = model.encode_batch(waveform.squeeze(0).unsqueeze(0))

    # embedding shape: (1, 1, 192) → flatten to (192,)
    embedding = embedding.squeeze().cpu().numpy().astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def get_mean_embedding(
    model: SpeakerRecognition,
    audio_paths: list[str | Path],
) -> np.ndarray:
    """
    Compute mean L2-normalized embedding across multiple audio files.
    Used for enrollment: averages per-file embeddings, then re-normalizes.
    """
    embeddings = [get_embedding(model, p) for p in audio_paths]
    mean_emb = np.mean(embeddings, axis=0).astype(np.float32)

    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    return mean_emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python embeddings.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"Loading model...")
    m = load_model()
    print(f"Extracting embedding from: {audio_file}")
    emb = get_embedding(m, audio_file)
    print(f"Embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")
    print(f"First 5 values: {emb[:5]}")
