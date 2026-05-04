from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bm25 import BM25Index
from .embeddings import make_embedder
from .ingest import documents_path
from .io import ensure_dir, load_documents, write_jsonl
from .schema import Chunk
from .text import chunk_documents


@dataclass
class HybridIndex:
    chunks: list[Chunk]
    bm25: BM25Index
    embeddings: np.ndarray
    metadata: dict[str, Any]


def chunks_path(index_dir: Path) -> Path:
    return index_dir / "chunks.jsonl"


def hybrid_index_path(index_dir: Path) -> Path:
    return index_dir / "hybrid_index.pkl"


def build_index(
    index_dir: Path,
    embedding: str = "hashing",
    embedding_model: str | None = None,
    embedding_dim: int = 384,
    chunk_tokens: int = 320,
    chunk_overlap: int = 64,
) -> HybridIndex:
    docs = load_documents(documents_path(index_dir))
    chunks = chunk_documents(docs, max_tokens=chunk_tokens, overlap=chunk_overlap)
    write_jsonl(chunks_path(index_dir), (chunk.to_json() for chunk in chunks))
    embedder = make_embedder(embedding, model=embedding_model, dim=embedding_dim)
    texts = [chunk.text for chunk in chunks]
    bm25 = BM25Index.build(texts)
    vectors = embedder.encode(texts) if texts else np.zeros((0, embedding_dim), dtype=np.float32)
    idx = HybridIndex(
        chunks=chunks,
        bm25=bm25,
        embeddings=vectors,
        metadata={
            "embedding": embedder.name,
            "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else embedding_dim,
            "document_count": len(docs),
            "chunk_count": len(chunks),
            "chunk_tokens": chunk_tokens,
            "chunk_overlap": chunk_overlap,
        },
    )
    save_index(index_dir, idx)
    return idx


def save_index(index_dir: Path, idx: HybridIndex) -> None:
    ensure_dir(index_dir)
    with hybrid_index_path(index_dir).open("wb") as handle:
        pickle.dump(idx, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(index_dir: Path) -> HybridIndex:
    path = hybrid_index_path(index_dir)
    if not path.exists():
        raise FileNotFoundError(f"No index found at {path}. Run `local-hybrid-ir build` first.")
    with path.open("rb") as handle:
        return pickle.load(handle)
