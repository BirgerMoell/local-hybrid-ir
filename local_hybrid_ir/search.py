from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .embeddings import make_embedder
from .index import HybridIndex


@dataclass
class SearchHit:
    score: float
    bm25_score: float
    dense_score: float
    chunk_id: str
    doc_id: str
    source: str
    kind: str
    title: str
    uri: str
    text: str
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "kind": self.kind,
            "title": self.title,
            "uri": self.uri,
            "text": self.text,
            "metadata": self.metadata,
        }


def search(
    idx: HybridIndex,
    query: str,
    k: int = 8,
    alpha: float = 0.35,
    embedding: str = "hashing",
    embedding_model: str | None = None,
    embedding_dim: int = 384,
    sources: set[str] | None = None,
    kinds: set[str] | None = None,
) -> list[SearchHit]:
    if not idx.chunks:
        return []
    alpha = min(1.0, max(0.0, alpha))
    bm25 = _normalize(idx.bm25.scores(query))
    embedder = make_embedder(embedding, model=embedding_model, dim=embedding_dim)
    q = embedder.encode([query])
    dense = np.sum(idx.embeddings * q[0].astype(np.float32), axis=1) if idx.embeddings.size else np.zeros(len(idx.chunks), dtype=np.float32)
    dense = _normalize(dense.astype(np.float32))
    combined = alpha * dense + (1.0 - alpha) * bm25
    mask = _mask(idx, sources=sources, kinds=kinds)
    if mask is not None:
        combined = combined.copy()
        combined[~mask] = -np.inf
    order = np.argsort(-combined)[: max(0, k)]
    hits: list[SearchHit] = []
    for i in order:
        if not np.isfinite(combined[i]):
            continue
        chunk = idx.chunks[int(i)]
        hits.append(
            SearchHit(
                score=float(combined[i]),
                bm25_score=float(bm25[i]),
                dense_score=float(dense[i]),
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                source=chunk.source,
                kind=chunk.kind,
                title=chunk.title,
                uri=chunk.uri,
                text=chunk.text,
                metadata=chunk.metadata,
            )
        )
    return hits


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi <= lo:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def _mask(idx: HybridIndex, sources: set[str] | None, kinds: set[str] | None) -> np.ndarray | None:
    if not sources and not kinds:
        return None
    keep = np.ones(len(idx.chunks), dtype=bool)
    if sources:
        keep &= np.asarray([chunk.source in sources for chunk in idx.chunks], dtype=bool)
    if kinds:
        keep &= np.asarray([chunk.kind in kinds for chunk in idx.chunks], dtype=bool)
    return keep
