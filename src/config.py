"""
Central configuration loader.

Reads config.toml from the project root and exposes typed attributes.
Import this module instead of hardcoding constants in other scripts.

Usage:
    from config import cfg

    wav_dir = cfg.paths.voxceleb_dir
    sr      = cfg.audio.sample_rate
"""

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # pip install tomli
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "TOML support not found. Run: pip install tomli"
        )
from dataclasses import dataclass
from pathlib import Path

# Project root is one level above src/
_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class _Paths:
    data_dir: Path
    voxceleb_dir: Path
    custom_dir: Path
    enrollment_dir: Path
    test_dir: Path
    db_dir: Path
    models_dir: Path
    results_dir: Path
    augmented_dir: Path


@dataclass(frozen=True)
class _Audio:
    sample_rate: int


@dataclass(frozen=True)
class _Dataset:
    num_speakers: int
    enrollment_samples_per_speaker: int
    random_seed: int


@dataclass(frozen=True)
class _HuggingFace:
    dataset_id: str
    split: str


@dataclass(frozen=True)
class _Model:
    source: str
    save_dir: Path
    embedding_dim: int


@dataclass(frozen=True)
class _Auth:
    threshold: float


@dataclass(frozen=True)
class Config:
    paths: _Paths
    audio: _Audio
    dataset: _Dataset
    huggingface: _HuggingFace
    model: _Model
    auth: _Auth


def _load() -> Config:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.toml not found at {_CONFIG_PATH}")

    with open(_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)

    def abs_path(rel: str) -> Path:
        return (_PROJECT_ROOT / rel).resolve()

    p = raw["paths"]
    return Config(
        paths=_Paths(
            data_dir=abs_path(p["data_dir"]),
            voxceleb_dir=abs_path(p["voxceleb_dir"]),
            custom_dir=abs_path(p["custom_dir"]),
            enrollment_dir=abs_path(p["enrollment_dir"]),
            test_dir=abs_path(p["test_dir"]),
            db_dir=abs_path(p["db_dir"]),
            models_dir=abs_path(p["models_dir"]),
            results_dir=abs_path(p["results_dir"]),
            augmented_dir=abs_path(p["augmented_dir"]),
        ),
        audio=_Audio(**raw["audio"]),
        dataset=_Dataset(**raw["dataset"]),
        huggingface=_HuggingFace(**raw["huggingface"]),
        model=_Model(
            source=raw["model"]["source"],
            save_dir=abs_path(raw["model"]["save_dir"]),
            embedding_dim=raw["model"]["embedding_dim"],
        ),
        auth=_Auth(**raw["auth"]),
    )


# Singleton – loaded once on first import
cfg: Config = _load()


if __name__ == "__main__":
    import pprint
    pprint.pprint(cfg)
