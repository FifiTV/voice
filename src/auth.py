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
