"""
Threshold selection via Equal Error Rate (EER).

Builds genuine and impostor score distributions from the test set,
finds the cosine similarity threshold at which FAR == FRR (EER),
and optionally saves it back to config.toml.

Usage:
    cd src
    python threshold.py                    # use data/test/, save threshold to config.toml
    python threshold.py --no-save          # just print, do not write config.toml
    python threshold.py --plot             # also save ROC and score distribution plots
    python threshold.py --n-impostors 5   # impostors per genuine sample (default: 1)
"""

import argparse
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from config import cfg
from database import get_collection, list_enrolled
from embeddings import load_model, get_embedding

TEST_DIR    = cfg.paths.test_dir
RESULTS_DIR = cfg.paths.results_dir
RANDOM_SEED = cfg.dataset.random_seed


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def build_trial_scores(
    model,
    collection,
    n_impostors_per_genuine: int = 1,
    max_genuine_per_speaker: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build arrays of genuine and impostor cosine similarity scores.

    Genuine trials:  pairs of test utterances from the SAME speaker vs. the
                     enrolled profile stored in ChromaDB.
    Impostor trials: utterances from speaker A tested against speaker B's profile.

    Returns:
        genuine_scores:  (N,) float array
        impostor_scores: (M,) float array
    """
    enrolled_ids = set(list_enrolled(collection))

    # Collect test files per speaker (only speakers that are enrolled)
    speaker_files: dict[str, list[Path]] = {}
    for spk_dir in sorted(TEST_DIR.iterdir()):
        if not spk_dir.is_dir() or spk_dir.name not in enrolled_ids:
            continue
        wavs = sorted(spk_dir.rglob("*.wav"))
        if wavs:
            speaker_files[spk_dir.name] = wavs

    if len(speaker_files) < 2:
        print("ERROR: Need at least 2 enrolled speakers with test files.")
        sys.exit(1)

    # Bulk-load all enrolled embeddings once — avoids per-ID get() calls that
    # fail in some chromadb Rust-backend versions (InternalError: Error finding id).
    enrolled_emb_cache = _load_enrolled_embeddings_bulk(collection)

    rng = random.Random(RANDOM_SEED)
    genuine_scores  = []
    impostor_scores = []

    speaker_ids = sorted(speaker_files.keys())
    print(f"Computing scores for {len(speaker_ids)} speakers...")

    audio_cache: dict[Path, np.ndarray] = {}

    def get_cached(path: Path) -> np.ndarray:
        if path not in audio_cache:
            audio_cache[path] = get_embedding(model, path)
        return audio_cache[path]

    for spk_id in speaker_ids:
        if spk_id not in enrolled_emb_cache:
            print(f"  [warn] {spk_id}: no enrolled embedding found, skipping")
            continue

        enrolled_emb = enrolled_emb_cache[spk_id]
        files = speaker_files[spk_id]
        rng.shuffle(files)
        selected = files[:max_genuine_per_speaker]

        for wav in selected:
            score = float(np.dot(get_cached(wav), enrolled_emb))
            genuine_scores.append(score)

        other_ids = [sid for sid in speaker_ids if sid != spk_id]
        impostor_speakers = rng.sample(
            other_ids, min(n_impostors_per_genuine * len(selected), len(other_ids))
        )
        for imp_id in impostor_speakers:
            imp_file = rng.choice(speaker_files[imp_id])
            score = float(np.dot(get_cached(imp_file), enrolled_emb))
            impostor_scores.append(score)

        print(f"  {spk_id}: {len(selected)} genuine, {len(impostor_speakers)} impostor")

    return np.array(genuine_scores), np.array(impostor_scores)


def _load_enrolled_embeddings_bulk(collection) -> dict[str, np.ndarray]:
    """
    Load ALL enrolled speaker embeddings in a single collection.get() call.

    Using a single bulk call instead of per-ID gets avoids a known issue in
    some chromadb Rust-backend versions where get(ids=[...], include=["embeddings"])
    raises InternalError: Error finding id even when the ID exists.
    """
    result = collection.get(include=["embeddings"])
    return {
        id_: np.array(emb, dtype=np.float32)
        for id_, emb in zip(result["ids"], result["embeddings"])
    }


# ---------------------------------------------------------------------------
# EER calculation
# ---------------------------------------------------------------------------

def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Equal Error Rate and the corresponding threshold.

    At EER:  FAR(threshold) == FRR(threshold)
      FAR = impostor samples accepted  / total impostors
      FRR = genuine samples rejected   / total genuines

    Returns:
        (eer, threshold)  both as fractions (0–1) and cosine similarity.
    """
    thresholds = np.linspace(
        min(genuine_scores.min(), impostor_scores.min()),
        max(genuine_scores.max(), impostor_scores.max()),
        num=1000,
    )

    far_list = []
    frr_list = []

    for thr in thresholds:
        far = np.mean(impostor_scores >= thr)   # impostors accepted
        frr = np.mean(genuine_scores  <  thr)   # genuines rejected
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # Find crossover via linear interpolation
    diff = far_arr - frr_arr
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # No crossover found – pick threshold minimizing |FAR - FRR|
        idx = np.argmin(np.abs(diff))
    else:
        idx = sign_changes[0]
        # Linear interpolation between idx and idx+1
        t_interp = interp1d(
            [diff[idx], diff[idx + 1]],
            [thresholds[idx], thresholds[idx + 1]],
        )
        thr_eer = float(t_interp(0.0))
        eer = float((far_arr[idx] + frr_arr[idx]) / 2)
        return eer, thr_eer

    eer = float((far_arr[idx] + frr_arr[idx]) / 2)
    return eer, float(thresholds[idx])


def compute_metrics_at_threshold(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
) -> dict:
    """Compute FAR, FRR, TAR, accuracy at a fixed threshold."""
    far = float(np.mean(impostor_scores >= threshold))
    frr = float(np.mean(genuine_scores  <  threshold))
    tar = 1.0 - frr  # True Accept Rate
    n_total = len(genuine_scores) + len(impostor_scores)
    tp = int(np.sum(genuine_scores  >= threshold))
    tn = int(np.sum(impostor_scores <  threshold))
    accuracy = (tp + tn) / n_total
    return {
        "threshold": threshold,
        "far":       far,
        "frr":       frr,
        "tar":       tar,
        "accuracy":  accuracy,
        "n_genuine":  len(genuine_scores),
        "n_impostor": len(impostor_scores),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    eer: float,
    eer_threshold: float,
    out_dir: Path = RESULTS_DIR / "plots",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed – skipping plots. Run: pip install matplotlib")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Score distributions
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(genuine_scores,  bins=60, alpha=0.6, label="Genuine",  color="steelblue")
    ax.hist(impostor_scores, bins=60, alpha=0.6, label="Impostor", color="tomato")
    ax.axvline(eer_threshold, color="black", linestyle="--", label=f"EER thr={eer_threshold:.3f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Score distributions  (EER={eer*100:.2f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=150)
    plt.close(fig)

    # ROC curve
    thresholds = np.linspace(
        min(genuine_scores.min(), impostor_scores.min()),
        max(genuine_scores.max(), impostor_scores.max()),
        num=500,
    )
    far_list = [np.mean(impostor_scores >= t) for t in thresholds]
    tar_list = [np.mean(genuine_scores  >= t) for t in thresholds]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(far_list, tar_list, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    ax.scatter([eer], [1 - eer], color="red", zorder=5, label=f"EER={eer*100:.2f}%")
    ax.set_xlabel("FAR (False Accept Rate)")
    ax.set_ylabel("TAR (True Accept Rate)")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Write threshold back to config.toml
# ---------------------------------------------------------------------------

def update_config_threshold(new_threshold: float) -> None:
    """Overwrite [auth] threshold in config.toml with the EER threshold."""
    config_path = Path(__file__).parent.parent / "config.toml"
    text = config_path.read_text(encoding="utf-8")

    import re
    pattern = r"(threshold\s*=\s*)[\d.+-]+"
    replacement = rf"\g<1>{new_threshold:.6f}"
    new_text, n = re.subn(pattern, replacement, text, count=1)

    if n == 0:
        print("[warn] Could not find 'threshold' in config.toml – update manually.")
        return

    config_path.write_text(new_text, encoding="utf-8")
    print(f"config.toml updated: [auth] threshold = {new_threshold:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EER-based threshold selection")
    p.add_argument("--n-impostors",  type=int,   default=1,
                   help="Impostor trials per genuine sample (default: 1)")
    p.add_argument("--max-genuine",  type=int,   default=50,
                   help="Max genuine utterances per speaker (default: 50)")
    p.add_argument("--no-save",      action="store_true",
                   help="Do not update config.toml")
    p.add_argument("--plot",         action="store_true",
                   help="Save ROC curve and score distribution plots")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("Loading ECAPA-TDNN model...")
    model = load_model()
    collection = get_collection()

    if collection.count() == 0:
        print("ERROR: No speakers enrolled. Run enroll.py first.")
        sys.exit(1)

    genuine, impostor = build_trial_scores(
        model, collection,
        n_impostors_per_genuine=args.n_impostors,
        max_genuine_per_speaker=args.max_genuine,
    )

    print(f"\nGenuine  scores: n={len(genuine)},  mean={genuine.mean():.4f}, std={genuine.std():.4f}")
    print(f"Impostor scores: n={len(impostor)}, mean={impostor.mean():.4f}, std={impostor.std():.4f}")

    eer, thr = compute_eer(genuine, impostor)
    metrics  = compute_metrics_at_threshold(genuine, impostor, thr)

    print(f"\n── EER threshold ───────────────────────────────")
    print(f"  EER       = {eer * 100:.2f}%")
    print(f"  Threshold = {thr:.6f}")
    print(f"  FAR       = {metrics['far'] * 100:.2f}%")
    print(f"  FRR       = {metrics['frr'] * 100:.2f}%")
    print(f"  Accuracy  = {metrics['accuracy'] * 100:.2f}%")
    print(f"────────────────────────────────────────────────")

    if args.plot:
        save_plots(genuine, impostor, eer, thr)

    if not args.no_save:
        update_config_threshold(thr)
