from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .text import tokenize


@dataclass
class BM25Index:
    documents: list[list[str]]
    doc_freq: dict[str, int]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, texts: Iterable[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        tokenized = [tokenize(text) for text in texts]
        df: dict[str, int] = {}
        for toks in tokenized:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        avgdl = sum(len(toks) for toks in tokenized) / max(1, len(tokenized))
        return cls(documents=tokenized, doc_freq=df, avgdl=avgdl, k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.documents), dtype=np.float32)
        terms = Counter(tokenize(query))
        if not terms or not self.documents:
            return scores
        n = len(self.documents)
        for term in terms:
            df = self.doc_freq.get(term, 0)
            if not df:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, toks in enumerate(self.documents):
                tf = toks.count(term)
                if not tf:
                    continue
                dl = len(toks)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * (tf * (self.k1 + 1.0) / denom)
        return scores
