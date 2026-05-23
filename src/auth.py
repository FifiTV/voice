"""
High-level authentication API.

Combines the embedding model (embeddings.py) with the ChromaDB store (database.py)
into two public operations:
  - verify(user_id, audio_path)   → 1-to-1 check: is this audio from user_id?
  - identify(audio_path)          → 1-to-N search: who is speaking?

The AuthSystem class holds the model and DB collection as singletons so they are
loaded once and reused across calls.

CLI usage:
    cd src
    python auth.py verify member_jan ../data/test/member_jan/utt_0005.wav
    python auth.py identify ../data/test/member_jan/utt_0005.wav
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from config import cfg
from database import get_collection, verify_speaker, identify_speaker
from embeddings import load_model, get_embedding
from chromadb import Collection
from speechbrain.inference.speaker import SpeakerRecognition


@dataclass
class VerifyResult:
    user_id: str
    accepted: bool
    score: float       # cosine similarity in [-1, 1]
    threshold: float

    def __str__(self) -> str:
        verdict = "ACCEPTED" if self.accepted else "REJECTED"
        return (
            f"[{verdict}] user_id={self.user_id}  "
            f"score={self.score:.4f}  threshold={self.threshold:.4f}"
        )


@dataclass
class IdentifyResult:
    user_id: str | None   # None if no speaker passes the threshold
    score: float
    threshold: float

    def __str__(self) -> str:
        if self.user_id is None:
            return f"[UNKNOWN]  score={self.score:.4f}  threshold={self.threshold:.4f}"
        return (
            f"[IDENTIFIED] user_id={self.user_id}  "
            f"score={self.score:.4f}  threshold={self.threshold:.4f}"
        )


class AuthSystem:
    """
    Stateful authentication system.

    Loads model and DB once; reuse the same instance for multiple auth calls.

    Example:
        auth = AuthSystem()
        result = auth.verify("member_jan", "recording.wav")
        print(result)
    """

    def __init__(
        self,
        threshold: float = cfg.auth.threshold,
        model_save_dir: Path = cfg.model.save_dir,
        db_path: Path = cfg.paths.db_dir,
    ) -> None:
        self.threshold = threshold
        print("Loading ECAPA-TDNN model...")
        self._model: SpeakerRecognition = load_model(model_save_dir)
        self._collection: Collection = get_collection(db_path)
        n = self._collection.count()
        print(f"Connected to DB – {n} speaker(s) enrolled.")

    def verify(self, user_id: str, audio_path: str | Path) -> VerifyResult:
        """
        Verify whether audio_path belongs to user_id.

        Raises KeyError if user_id is not enrolled.
        """
        embedding = get_embedding(self._model, audio_path)
        accepted, score = verify_speaker(
            self._collection, user_id, embedding, self.threshold
        )
        return VerifyResult(
            user_id=user_id,
            accepted=accepted,
            score=score,
            threshold=self.threshold,
        )

    def identify(self, audio_path: str | Path) -> IdentifyResult:
        """
        Identify the speaker in audio_path by nearest-neighbour search.
        Returns user_id=None if no enrolled speaker exceeds the threshold.
        """
        embedding = get_embedding(self._model, audio_path)
        user_id, score = identify_speaker(
            self._collection, embedding, self.threshold
        )
        return IdentifyResult(user_id=user_id, score=score, threshold=self.threshold)

    def top_matches(
        self, audio_path: str | Path, n: int = 5
    ) -> list[tuple[str, float]]:
        """Return top-n (user_id, similarity) pairs sorted by score descending."""
        embedding = get_embedding(self._model, audio_path)
        return identify_speaker(self._collection, embedding, self.threshold, top_n=n)

    def demo(self, audio_path: str | Path, claimed_id: str | None = None) -> None:
        """Pretty-print verification/identification results for live demo."""
        import os
        audio_path = Path(audio_path)
        duration = _audio_duration(audio_path)

        print()
        print("=" * 56)
        print("  VOICE AUTHENTICATION DEMO")
        print("=" * 56)
        print(f"  File      : {audio_path.name}")
        print(f"  Duration  : {duration:.2f} s")
        print(f"  Threshold : {self.threshold:.4f}")
        print("-" * 56)

        # 1-to-1 verification
        if claimed_id:
            print(f"\n  [VERIFY]  claimed identity: {claimed_id}")
            try:
                result = self.verify(claimed_id, audio_path)
                _print_score_bar(result.score, self.threshold)
                verdict = "ACCEPTED" if result.accepted else "REJECTED"
                print(f"  Result    : {verdict}")
            except KeyError:
                print(f"  ERROR: '{claimed_id}' is not enrolled.")

        # 1-to-N identification — top 5
        print("\n  [IDENTIFY]  top-5 candidates")
        hits = self.top_matches(audio_path, n=5)
        if not hits:
            print("  No speakers in database.")
        else:
            for rank, (uid, score) in enumerate(hits, 1):
                tag = "<-- match" if score >= self.threshold else ""
                bar = _score_bar(score)
                print(f"  {rank}. {uid:<20s}  {bar}  {score:.4f}  {tag}")

        print("=" * 56)
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_bar(score: float, width: int = 20) -> str:
    filled = max(0, min(width, round(score * width)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_score_bar(score: float, threshold: float, width: int = 20) -> None:
    bar = _score_bar(score, width)
    thr_pos = max(0, min(width, round(threshold * width)))
    marker = " " * (thr_pos + 1) + "^threshold"
    print(f"  Score     : {score:.4f}  {bar}")
    print(f"             {marker}")


def _audio_duration(audio_path: Path) -> float:
    try:
        import torchaudio
        info = torchaudio.info(str(audio_path))
        return info.num_frames / info.sample_rate
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice authentication CLI")
    parser.add_argument(
        "--threshold", type=float, default=cfg.auth.threshold,
        help=f"Cosine similarity threshold (default: {cfg.auth.threshold})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="1-to-1 verification")
    p_verify.add_argument("user_id", help="Enrolled speaker ID to verify against")
    p_verify.add_argument("audio_path", type=Path, help="Audio file to test")

    p_identify = sub.add_parser("identify", help="1-to-N identification")
    p_identify.add_argument("audio_path", type=Path, help="Audio file to identify")

    p_demo = sub.add_parser("demo", help="Live demo: pretty-print top matches + optional verify")
    p_demo.add_argument("audio_path", type=Path, help="Audio file (WAV, MP3, M4A, OGG, …)")
    p_demo.add_argument("--user-id", type=str, default=None,
                        help="Claimed speaker ID for 1-to-1 verification (optional)")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    auth = AuthSystem(threshold=args.threshold)

    if args.command == "verify":
        try:
            result = auth.verify(args.user_id, args.audio_path)
        except KeyError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        print(result)
        sys.exit(0 if result.accepted else 1)

    elif args.command == "identify":
        result = auth.identify(args.audio_path)
        print(result)
        sys.exit(0 if result.user_id is not None else 1)

    elif args.command == "demo":
        auth.demo(args.audio_path, claimed_id=args.user_id)
