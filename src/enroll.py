"""
Enrollment pipeline: reads audio files from data/enrollment/ (or a custom path),
extracts ECAPA-TDNN embeddings, and stores mean embeddings in ChromaDB.

Usage:
    # Enroll all speakers from data/enrollment/
    cd src
    python enroll.py

    # Enroll a single speaker from a specific folder
    python enroll.py --speaker-id member_jan --audio-dir ../data/custom/member_jan

    # Re-enroll (overwrite) all speakers
    python enroll.py --overwrite

    # List enrolled speakers
    python enroll.py --list
"""

import argparse
import sys
from pathlib import Path

from config import cfg
from database import get_collection, enroll_user, list_enrolled, remove_user
from embeddings import load_model, get_mean_embedding

ENROLLMENT_DIR = cfg.paths.enrollment_dir
SUPPORTED_EXT  = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def enroll_all(overwrite: bool = False, exclude_prefix: str | None = None) -> None:
    """Enroll every speaker found in data/enrollment/<speaker_id>/."""
    if not ENROLLMENT_DIR.exists():
        print(f"ERROR: Enrollment directory not found: {ENROLLMENT_DIR}")
        sys.exit(1)

    speaker_dirs = sorted([d for d in ENROLLMENT_DIR.iterdir() if d.is_dir()])
    if exclude_prefix:
        before = len(speaker_dirs)
        speaker_dirs = [d for d in speaker_dirs if not d.name.startswith(exclude_prefix)]
        print(f"Excluding {before - len(speaker_dirs)} speaker(s) with prefix '{exclude_prefix}'.")
    if not speaker_dirs:
        print(f"No speaker folders found in {ENROLLMENT_DIR}")
        sys.exit(1)

    print(f"Loading ECAPA-TDNN model...")
    model = load_model()
    collection = get_collection()

    already_enrolled = set(list_enrolled(collection))
    enrolled_count = 0
    skipped_count = 0

    for spk_dir in speaker_dirs:
        speaker_id = spk_dir.name

        if speaker_id in already_enrolled and not overwrite:
            print(f"  [skip] {speaker_id} already enrolled (use --overwrite to re-enroll)")
            skipped_count += 1
            continue

        audio_files = [
            f for f in sorted(spk_dir.iterdir())
            if f.suffix.lower() in SUPPORTED_EXT
        ]

        if not audio_files:
            print(f"  [warn] {speaker_id}: no audio files found, skipping")
            continue

        print(f"  [{enrolled_count + 1}] {speaker_id}: {len(audio_files)} file(s) → ", end="", flush=True)

        try:
            mean_emb = get_mean_embedding(model, audio_files)
            enroll_user(
                collection,
                user_id=speaker_id,
                mean_embedding=mean_emb,
                metadata={"n_enrollment_files": len(audio_files)},
            )
            print("enrolled")
            enrolled_count += 1
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone. {enrolled_count} enrolled, {skipped_count} skipped.")
    print(f"Total speakers in DB: {collection.count()}")


def enroll_single(speaker_id: str, audio_dir: Path, overwrite: bool = False) -> None:
    """Enroll a single speaker from a specific directory."""
    audio_files = [
        f for f in sorted(audio_dir.iterdir())
        if f.suffix.lower() in SUPPORTED_EXT
    ]

    if not audio_files:
        print(f"ERROR: No audio files found in {audio_dir}")
        sys.exit(1)

    collection = get_collection()
    already_enrolled = set(list_enrolled(collection))

    if speaker_id in already_enrolled and not overwrite:
        print(f"Speaker '{speaker_id}' already enrolled. Use --overwrite to replace.")
        sys.exit(0)

    print(f"Loading ECAPA-TDNN model...")
    model = load_model()

    print(f"Enrolling '{speaker_id}' from {len(audio_files)} file(s)...")
    mean_emb = get_mean_embedding(model, audio_files)
    enroll_user(
        collection,
        user_id=speaker_id,
        mean_embedding=mean_emb,
        metadata={"n_enrollment_files": len(audio_files)},
    )
    print(f"Done. Speaker '{speaker_id}' enrolled successfully.")
    print(f"Total speakers in DB: {collection.count()}")


def delete_speaker(speaker_id: str) -> None:
    collection = get_collection()
    enrolled = set(list_enrolled(collection))
    if speaker_id not in enrolled:
        print(f"Speaker '{speaker_id}' not found in database.")
        sys.exit(1)
    remove_user(collection, speaker_id)


def delete_prefix(prefix: str, yes: bool = False) -> None:
    collection = get_collection()
    matches = [sid for sid in list_enrolled(collection) if sid.startswith(prefix)]
    if not matches:
        print(f"No enrolled speakers with prefix '{prefix}'.")
        return
    if not yes:
        ans = input(f"Delete {len(matches)} speaker(s) with prefix '{prefix}'? [y/N] ")
        if ans.strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)
    for sid in matches:
        collection.delete(ids=[sid])
    print(f"Removed {len(matches)} speaker(s) with prefix '{prefix}'.")


def delete_all(yes: bool = False) -> None:
    collection = get_collection()
    enrolled = list_enrolled(collection)
    if not enrolled:
        print("Database is already empty.")
        return
    if not yes:
        ans = input(f"Delete ALL {len(enrolled)} speakers from the database? [y/N] ")
        if ans.strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)
    for sid in enrolled:
        collection.delete(ids=[sid])
    print(f"Removed {len(enrolled)} speaker(s). Database is now empty.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll speakers into ChromaDB")
    parser.add_argument(
        "--speaker-id", type=str, default=None,
        help="Enroll a single speaker (requires --audio-dir)",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=None,
        help="Path to folder with audio files for --speaker-id",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-enroll speakers already in the database",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List enrolled speakers and exit",
    )
    parser.add_argument(
        "--exclude", type=str, metavar="PREFIX",
        help="Skip speakers whose ID starts with PREFIX when enrolling (e.g. ls_)",
    )
    parser.add_argument(
        "--delete", type=str, metavar="SPEAKER_ID",
        help="Remove a single speaker from the database",
    )
    parser.add_argument(
        "--delete-prefix", type=str, metavar="PREFIX",
        help="Remove all speakers whose ID starts with PREFIX (e.g. ls_)",
    )
    parser.add_argument(
        "--delete-all", action="store_true",
        help="Remove all speakers from the database",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip confirmation prompt for --delete-prefix and --delete-all",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        collection = get_collection()
        enrolled = list_enrolled(collection)
        print(f"Enrolled speakers ({len(enrolled)}):")
        for sid in sorted(enrolled):
            print(f"  {sid}")
        sys.exit(0)

    if args.delete:
        delete_speaker(args.delete)
        sys.exit(0)

    if args.delete_prefix:
        delete_prefix(args.delete_prefix, yes=args.yes)
        sys.exit(0)

    if args.delete_all:
        delete_all(yes=args.yes)
        sys.exit(0)

    if args.speaker_id:
        if not args.audio_dir:
            print("ERROR: --audio-dir is required when using --speaker-id")
            sys.exit(1)
        enroll_single(args.speaker_id, args.audio_dir, overwrite=args.overwrite)
    else:
        enroll_all(overwrite=args.overwrite, exclude_prefix=args.exclude)
