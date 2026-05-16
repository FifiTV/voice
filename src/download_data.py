"""
Dataset acquisition scripts for the voice authentication project.

Available commands:
  download-hf     – stream selected speakers from HuggingFace (no full download)
  organize        – organize already-downloaded VoxCeleb1 zip into project layout
  split           – split data/voxceleb1/ into enrollment / test subsets
  download-custom – download YouTube audio for group members via yt-dlp
"""

import argparse
import csv
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio

from config import cfg

import os
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

# Shortcuts from config
_p = cfg.paths
DATA_DIR        = _p.data_dir
VOXCELEB_DIR    = _p.voxceleb_dir
CUSTOM_DIR      = _p.custom_dir
ENROLLMENT_DIR  = _p.enrollment_dir
TEST_DIR        = _p.test_dir

SAMPLE_RATE                    = cfg.audio.sample_rate
NUM_SPEAKERS                   = cfg.dataset.num_speakers
ENROLLMENT_SAMPLES_PER_SPEAKER = cfg.dataset.enrollment_samples_per_speaker
RANDOM_SEED                    = cfg.dataset.random_seed
HF_DATASET_ID                  = cfg.huggingface.dataset_id
HF_DATASET_SPLIT               = cfg.huggingface.split


# ---------------------------------------------------------------------------
# HuggingFace streaming download
# ---------------------------------------------------------------------------

def download_from_hf(
    n_speakers: int = NUM_SPEAKERS,
    min_utterances: int = ENROLLMENT_SAMPLES_PER_SPEAKER + 10,
    dataset_id: str = HF_DATASET_ID,
    hf_split: str = HF_DATASET_SPLIT,
) -> None:
    """
    Stream VoxCeleb1 from HuggingFace and save only the selected speakers.

    Uses datasets streaming mode – audio is downloaded on-the-fly, so only
    the utterances we actually keep are written to disk (~few GB instead of 39 GB).

    Args:
        n_speakers:       how many speakers to collect
        min_utterances:   skip speakers with fewer utterances than this
        dataset_id:       HuggingFace dataset repo id
        hf_split:         dataset split to stream from
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: pip install datasets")
        sys.exit(1)

    VOXCELEB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {dataset_id} (split={hf_split}) ...")
    print("Audio is downloaded on-demand – only kept utterances are saved.\n")

    # ds = load_dataset(dataset_id, split=hf_split, streaming=True, trust_remote_code=True)
    ds = load_dataset(dataset_id, split=hf_split, streaming=True)

    # Buffer: speaker_id -> list of (array, sampling_rate)
    buffer: dict[str, list[tuple]] = defaultdict(list)
    collected_speakers: set[str] = set()

    for sample in ds:
        speaker_id: str = sample.get("speaker_id") or sample.get("id") or sample.get("label")
        if speaker_id is None:
            # Try to infer speaker id from the audio path metadata
            audio_info = sample.get("audio", {})
            path_str = audio_info.get("path", "")
            parts = Path(path_str).parts
            # VoxCeleb layout: .../<id10xxx>/<video>/<utt>.wav
            speaker_id = parts[-3] if len(parts) >= 3 else path_str

        speaker_id = str(speaker_id)

        if speaker_id in collected_speakers:
            continue

        audio_info = sample.get("audio", {})
        array = audio_info.get("array")
        sr = audio_info.get("sampling_rate", SAMPLE_RATE)

        if array is None:
            continue

        buffer[speaker_id].append((array, sr))

        if len(buffer[speaker_id]) >= min_utterances:
            _save_speaker_from_buffer(speaker_id, buffer.pop(speaker_id))
            collected_speakers.add(speaker_id)
            print(f"  [{len(collected_speakers):3d}/{n_speakers}] saved speaker {speaker_id}")

        if len(collected_speakers) >= n_speakers:
            break

    if len(collected_speakers) < n_speakers:
        for speaker_id, utts in buffer.items():
            if len(utts) >= ENROLLMENT_SAMPLES_PER_SPEAKER + 1:
                _save_speaker_from_buffer(speaker_id, utts)
                collected_speakers.add(speaker_id)
                print(f"  [{len(collected_speakers):3d}/{n_speakers}] saved speaker {speaker_id} (partial)")
            if len(collected_speakers) >= n_speakers:
                break

    print(f"\nDone. {len(collected_speakers)} speakers saved to {VOXCELEB_DIR}")
    split_enrollment_test()
    save_speaker_list()


def _save_speaker_from_buffer(speaker_id: str, utterances: list[tuple]) -> None:
    """Write buffered utterances for one speaker to data/voxceleb1/<speaker_id>/."""
    spk_dir = VOXCELEB_DIR / speaker_id
    spk_dir.mkdir(parents=True, exist_ok=True)

    for idx, (array, sr) in enumerate(utterances):
        out_path = spk_dir / f"utt_{idx:04d}.wav"
        if out_path.exists():
            continue

        waveform = torch.tensor(array, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        torchaudio.save(str(out_path), waveform, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# VoxCeleb1 local organisation (for users who downloaded the full zip)
# ---------------------------------------------------------------------------

def organize_voxceleb(raw_vox_root: Path, n_speakers: int = NUM_SPEAKERS) -> None:
    """
    Select n_speakers from a locally extracted VoxCeleb1 dev set and copy
    their WAV files into data/voxceleb1/<speaker_id>/.

    Expected raw layout (after extracting vox1_dev_wav.zip):
        <raw_vox_root>/wav/<id10xxx>/<video_id>/<utt_id>.wav
    """
    wav_root = raw_vox_root / "wav"
    if not wav_root.exists():
        wav_root = raw_vox_root  # some releases unzip directly as id10xxx/ folders

    all_speakers = sorted([p for p in wav_root.iterdir() if p.is_dir()])
    if len(all_speakers) < n_speakers:
        raise ValueError(
            f"Found only {len(all_speakers)} speakers in {wav_root}, need {n_speakers}."
        )

    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(all_speakers, n_speakers)

    print(f"Copying {n_speakers} speakers to {VOXCELEB_DIR} ...")
    for spk_dir in selected:
        dest = VOXCELEB_DIR / spk_dir.name
        if dest.exists():
            print(f"  [skip] {spk_dir.name} already exists")
            continue
        shutil.copytree(spk_dir, dest)
        print(f"  [ok] {spk_dir.name}")

    print("Done.")


# ---------------------------------------------------------------------------
# Enrollment / test split
# ---------------------------------------------------------------------------

def split_enrollment_test(speaker_root: Path = VOXCELEB_DIR) -> None:
    """
    For each speaker in data/voxceleb1/ split WAV files into:
      data/enrollment/<speaker_id>/  – first ENROLLMENT_SAMPLES_PER_SPEAKER files
      data/test/<speaker_id>/        – remaining files (never used during training)

    Creates symlinks to avoid data duplication (copies on Windows if symlinks unavailable).
    """
    rng = random.Random(RANDOM_SEED)
    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    for spk_dir in sorted(speaker_root.iterdir()):
        if not spk_dir.is_dir():
            continue

        wavs = sorted(spk_dir.rglob("*.wav"))
        rng.shuffle(wavs)

        if len(wavs) < ENROLLMENT_SAMPLES_PER_SPEAKER + 1:
            print(f"[warn] {spk_dir.name}: only {len(wavs)} files, skipping")
            continue

        enroll_wavs = wavs[:ENROLLMENT_SAMPLES_PER_SPEAKER]
        test_wavs = wavs[ENROLLMENT_SAMPLES_PER_SPEAKER:]

        for split_dir_root, files in [
            (ENROLLMENT_DIR, enroll_wavs),
            (TEST_DIR, test_wavs),
        ]:
            split_dir = split_dir_root / spk_dir.name
            split_dir.mkdir(parents=True, exist_ok=True)
            for wav in files:
                link = split_dir / wav.name
                if not link.exists():
                    try:
                        link.symlink_to(wav.resolve())
                    except OSError:
                        shutil.copy2(wav, link)

        print(f"  {spk_dir.name}: {len(enroll_wavs)} enroll / {len(test_wavs)} test")

    print("Enrollment/test split complete.")


def save_speaker_list(
    speaker_root: Path = VOXCELEB_DIR,
    out_csv: Path = DATA_DIR / "speakers.csv",
) -> None:
    """Write CSV: speaker_id, n_enrollment_files, n_test_files."""
    rows = []
    for spk_dir in sorted(speaker_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        enroll_dir = ENROLLMENT_DIR / spk_dir.name
        test_dir = TEST_DIR / spk_dir.name
        n_enroll = len(list(enroll_dir.rglob("*.wav"))) if enroll_dir.exists() else 0
        n_test = len(list(test_dir.rglob("*.wav"))) if test_dir.exists() else 0
        rows.append({"speaker_id": spk_dir.name, "n_enrollment": n_enroll, "n_test": n_test})

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker_id", "n_enrollment", "n_test"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Speaker list saved to {out_csv} ({len(rows)} speakers)")


# ---------------------------------------------------------------------------
# Custom recordings via yt-dlp (group members)
# ---------------------------------------------------------------------------

def _check_tool(name: str) -> None:
    if shutil.which(name) is None:
        print(f"ERROR: '{name}' not found in PATH. Install it first.")
        sys.exit(1)


def download_custom(
    youtube_urls: list[str],
    speaker_id: str,
    max_duration_s: int = 120,
) -> None:
    """
    Download YouTube video(s) for a group member, extract audio,
    resample to 16kHz mono WAV, and save to data/custom/<speaker_id>/.
    """
    _check_tool("yt-dlp")
    _check_tool("ffmpeg")

    out_dir = CUSTOM_DIR / speaker_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    for url in youtube_urls:
        print(f"\nDownloading: {url}")
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--match-filter", f"duration < {max_duration_s}",
            "--output", str(tmp_dir / "%(id)s.%(ext)s"),
            url,
        ]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  [warn] yt-dlp failed for {url}")

    for wav_file in sorted(tmp_dir.glob("*.wav")):
        out_path = out_dir / wav_file.name
        _resample_to_16k(wav_file, out_path)
        wav_file.unlink()

    tmp_dir.rmdir()
    print(f"\nCustom recordings saved to {out_dir}")


def _resample_to_16k(src: Path, dst: Path) -> None:
    waveform, sr = torchaudio.load(str(src))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    torchaudio.save(str(dst), waveform, SAMPLE_RATE)
    print(f"  [ok] {dst.name}  ({sr}Hz → {SAMPLE_RATE}Hz)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice auth – data acquisition")
    sub = parser.add_subparsers(dest="command", required=True)

    p_hf = sub.add_parser(
        "download-hf",
        help="Stream VoxCeleb1 from HuggingFace (recommended – no full download)",
    )
    p_hf.add_argument("--n-speakers", type=int, default=NUM_SPEAKERS)
    p_hf.add_argument("--min-utterances", type=int, default=ENROLLMENT_SAMPLES_PER_SPEAKER + 10)
    p_hf.add_argument("--dataset-id", type=str, default=HF_DATASET_ID)
    p_hf.add_argument("--split", type=str, default=HF_DATASET_SPLIT)

    p_org = sub.add_parser("organize", help="Organize locally extracted VoxCeleb1")
    p_org.add_argument("raw_vox_root", type=Path)
    p_org.add_argument("--n-speakers", type=int, default=NUM_SPEAKERS)

    sub.add_parser("split", help="Split data/voxceleb1/ into enrollment and test")

    p_dl = sub.add_parser("download-custom", help="Download YouTube audio for a group member")
    p_dl.add_argument("speaker_id")
    p_dl.add_argument("urls", nargs="+")
    p_dl.add_argument("--max-duration", type=int, default=120)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "download-hf":
        download_from_hf(
            n_speakers=args.n_speakers,
            min_utterances=args.min_utterances,
            dataset_id=args.dataset_id,
            hf_split=args.split,
        )
    elif args.command == "organize":
        organize_voxceleb(args.raw_vox_root, args.n_speakers)
        split_enrollment_test()
        save_speaker_list()
    elif args.command == "split":
        split_enrollment_test()
        save_speaker_list()
    elif args.command == "download-custom":
        download_custom(args.urls, args.speaker_id, args.max_duration)
