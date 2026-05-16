"""
Dataset acquisition scripts for the voice authentication project.

Two modes:
  1. organize_voxceleb  – organizes already-downloaded VoxCeleb1 into the
                          project's data/ layout and selects 100 speakers.
  2. download_custom    – uses yt-dlp to download YouTube videos for group
                          members, extracts audio, resamples to 16kHz mono WAV.

VoxCeleb1 must be downloaded manually (requires VGG registration):
  https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
  Download: vox1_dev_wav.zip  (dev set, ~39 GB)
"""

import argparse
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torchaudio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
VOXCELEB_DIR = DATA_DIR / "voxceleb1"
CUSTOM_DIR = DATA_DIR / "custom"
ENROLLMENT_DIR = DATA_DIR / "enrollment"
TEST_DIR = DATA_DIR / "test"

SAMPLE_RATE = 16000
NUM_SPEAKERS = 100
ENROLLMENT_SAMPLES_PER_SPEAKER = 5  # files kept for enrollment
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# VoxCeleb1 organisation
# ---------------------------------------------------------------------------

def organize_voxceleb(raw_vox_root: Path, n_speakers: int = NUM_SPEAKERS) -> None:
    """
    Select n_speakers from VoxCeleb1 dev set and copy/symlink their WAV files
    into data/voxceleb1/<speaker_id>/.

    Expected raw layout (after extracting vox1_dev_wav.zip):
        <raw_vox_root>/wav/<id10xxx>/<video_id>/<utt_id>.wav

    Output layout:
        data/voxceleb1/<id10xxx>/<video_id>/<utt_id>.wav
    """
    wav_root = raw_vox_root / "wav"
    if not wav_root.exists():
        # Some releases unzip directly as id10xxx/ folders
        wav_root = raw_vox_root

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


def split_enrollment_test(speaker_root: Path = VOXCELEB_DIR) -> None:
    """
    For each speaker in data/voxceleb1/, shuffle their WAV files and split into:
      data/enrollment/<speaker_id>/  – first ENROLLMENT_SAMPLES_PER_SPEAKER files
      data/test/<speaker_id>/        – remaining files

    Creates symlinks (no data duplication).
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
            print(f"[warn] {spk_dir.name}: only {len(wavs)} files, skipping split")
            continue

        enroll_wavs = wavs[:ENROLLMENT_SAMPLES_PER_SPEAKER]
        test_wavs = wavs[ENROLLMENT_SAMPLES_PER_SPEAKER:]

        for split_name, files in [("enrollment", enroll_wavs), ("test", test_wavs)]:
            split_dir = (ENROLLMENT_DIR if split_name == "enrollment" else TEST_DIR) / spk_dir.name
            split_dir.mkdir(parents=True, exist_ok=True)
            for wav in files:
                link = split_dir / wav.name
                if not link.exists():
                    link.symlink_to(wav.resolve())

        print(f"  {spk_dir.name}: {len(enroll_wavs)} enroll / {len(test_wavs)} test")

    print("Enrollment/test split complete.")


def save_speaker_list(speaker_root: Path = VOXCELEB_DIR, out_csv: Path = DATA_DIR / "speakers.csv") -> None:
    """Write CSV: speaker_id, n_enrollment_files, n_test_files."""
    rows = []
    for spk_dir in sorted(speaker_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        n_enroll = len(list((ENROLLMENT_DIR / spk_dir.name).rglob("*.wav"))) if (ENROLLMENT_DIR / spk_dir.name).exists() else 0
        n_test = len(list((TEST_DIR / spk_dir.name).rglob("*.wav"))) if (TEST_DIR / spk_dir.name).exists() else 0
        rows.append({"speaker_id": spk_dir.name, "n_enrollment": n_enroll, "n_test": n_test})

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker_id", "n_enrollment", "n_test"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Speaker list saved to {out_csv} ({len(rows)} speakers)")


# ---------------------------------------------------------------------------
# Custom recordings via yt-dlp
# ---------------------------------------------------------------------------

def _check_tool(name: str) -> None:
    if shutil.which(name) is None:
        print(f"ERROR: '{name}' not found in PATH. Install it first.")
        sys.exit(1)


def download_custom(youtube_urls: list[str], speaker_id: str, max_duration_s: int = 120) -> None:
    """
    Download YouTube video(s) for a group member, extract audio,
    resample to 16kHz mono WAV, and save to data/custom/<speaker_id>/.

    Args:
        youtube_urls: list of YouTube video URLs
        speaker_id:   unique identifier, e.g. 'member_jan_kowalski'
        max_duration_s: skip videos longer than this (seconds)
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

    # Resample all downloaded WAVs to 16kHz mono
    for wav_file in sorted(tmp_dir.glob("*.wav")):
        out_path = out_dir / wav_file.name
        _resample_to_16k(wav_file, out_path)
        wav_file.unlink()

    tmp_dir.rmdir()
    print(f"\nCustom recordings saved to {out_dir}")


def _resample_to_16k(src: Path, dst: Path) -> None:
    """Resample audio to 16kHz mono WAV using torchaudio."""
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

    # organize voxceleb
    p_org = sub.add_parser("organize", help="Organize VoxCeleb1 into project layout")
    p_org.add_argument("raw_vox_root", type=Path, help="Path to extracted vox1_dev_wav directory")
    p_org.add_argument("--n-speakers", type=int, default=NUM_SPEAKERS)

    # split enrollment/test
    sub.add_parser("split", help="Split speakers into enrollment and test subsets")

    # download custom member recordings
    p_dl = sub.add_parser("download-custom", help="Download YouTube audio for a group member")
    p_dl.add_argument("speaker_id", help="Speaker identifier, e.g. 'member_jan_kowalski'")
    p_dl.add_argument("urls", nargs="+", help="YouTube URLs")
    p_dl.add_argument("--max-duration", type=int, default=120, help="Skip videos longer than N seconds")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "organize":
        organize_voxceleb(args.raw_vox_root, args.n_speakers)
        split_enrollment_test()
        save_speaker_list()

    elif args.command == "split":
        split_enrollment_test()
        save_speaker_list()

    elif args.command == "download-custom":
        download_custom(args.urls, args.speaker_id, args.max_duration)
