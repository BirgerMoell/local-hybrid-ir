from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


@dataclass
class HashingDenseEmbedder:
    """A deterministic local dense embedder.

    This is not as semantically strong as a trained embedding model, but it is
    useful for teaching because it has no network dependency and makes vectors
    inspectable.
    """

    dim: int = 384
    name: str = "hashing-dense-v1"

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            features = tokenize(text)
            features.extend(_char_ngrams(text.lower(), 4))
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8", "ignore"), digest_size=8).digest()
                value = int.from_bytes(digest, "little", signed=False)
                col = value % self.dim
                sign = 1.0 if (value >> 63) == 0 else -1.0
                matrix[row, col] += sign
            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedder:
    def __init__(self, model: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.name = f"sentence-transformers:{model}"
        self.batch_size = batch_size
        self.model = SentenceTransformer(model)
        self.dim = int(self.model.get_sentence_embedding_dimension() or 0)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def make_embedder(kind: str, model: str | None = None, dim: int = 384) -> Embedder:
    if kind == "hashing":
        return HashingDenseEmbedder(dim=dim)
    if kind == "sentence-transformers":
        if not model:
            raise ValueError("--embedding-model is required with --embedding sentence-transformers")
        return SentenceTransformerEmbedder(model)
    raise ValueError(f"Unknown embedding kind: {kind}")


def _char_ngrams(text: str, n: int) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    if len(clean) <= n:
        return [clean]
    return [clean[i : i + n] for i in range(0, len(clean) - n + 1, 2)]
