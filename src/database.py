"""
ChromaDB wrapper for the voice authentication system.

Stores one mean embedding per enrolled speaker.
All embeddings are L2-normalized, so cosine similarity == dot product.

Collection metric: cosine  →  ChromaDB returns distance = 1 - cosine_similarity.
"""

import numpy as np
import chromadb
from chromadb import Collection
from pathlib import Path

from config import cfg

COLLECTION_NAME = "speakers"


def get_collection(db_path: Path = cfg.paths.db_dir) -> Collection:
    """Open (or create) the persistent ChromaDB collection."""
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def enroll_user(
    collection: Collection,
    user_id: str,
    mean_embedding: np.ndarray,
    metadata: dict | None = None,
) -> None:
    """
    Store or overwrite a speaker's profile embedding in ChromaDB.

    Args:
        collection:      ChromaDB collection returned by get_collection()
        user_id:         unique speaker identifier (e.g. 'id10001' or 'member_jan')
        mean_embedding:  L2-normalized float32 array of shape (embedding_dim,)
        metadata:        optional extra fields stored alongside the embedding
    """
    meta = metadata or {}
    meta["user_id"] = user_id

    existing = collection.get(ids=[user_id])
    if existing["ids"]:
        collection.update(
            ids=[user_id],
            embeddings=[mean_embedding.tolist()],
            metadatas=[meta],
        )
    else:
        collection.add(
            ids=[user_id],
            embeddings=[mean_embedding.tolist()],
            metadatas=[meta],
        )


def verify_speaker(
    collection: Collection,
    user_id: str,
    query_embedding: np.ndarray,
    threshold: float = cfg.auth.threshold,
) -> tuple[bool, float]:
    """
    Verify whether a query embedding belongs to user_id.

    Returns:
        (accepted, cosine_similarity)
        accepted is True if similarity >= threshold.
    """
    # Filter by metadata (where=) instead of by id= to avoid a bug in some
    # chromadb Rust-backend versions where get(ids=..., include=["embeddings"])
    # raises InternalError: Error finding id.
    result = collection.get(
        where={"user_id": user_id},
        include=["embeddings"],
    )
    if not result["ids"]:
        raise KeyError(f"Speaker '{user_id}' not found in database.")

    enrolled_emb = np.array(result["embeddings"][0], dtype=np.float32)
    similarity = float(np.dot(query_embedding, enrolled_emb))
    return similarity >= threshold, similarity


def identify_speaker(
    collection: Collection,
    query_embedding: np.ndarray,
    threshold: float = cfg.auth.threshold,
) -> tuple[str | None, float]:
    """
    Identify the closest matching speaker for a query embedding.

    Returns:
        (user_id, cosine_similarity) if similarity >= threshold, else (None, similarity).
    """
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=1,
        include=["distances", "metadatas"],
    )

    if not result["ids"][0]:
        return None, 0.0

    # ChromaDB cosine distance = 1 - cosine_similarity
    distance = result["distances"][0][0]
    similarity = 1.0 - distance
    user_id = result["ids"][0][0]

    if similarity >= threshold:
        return user_id, similarity
    return None, similarity


def list_enrolled(collection: Collection) -> list[str]:
    """Return list of all enrolled speaker IDs."""
    return collection.get(include=[])["ids"]


def remove_user(collection: Collection, user_id: str) -> None:
    """Delete a speaker's profile from the database."""
    collection.delete(ids=[user_id])
    print(f"Removed speaker '{user_id}' from database.")
