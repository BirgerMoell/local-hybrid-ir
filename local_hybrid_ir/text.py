from __future__ import annotations

import html
import re
from hashlib import sha1
from typing import Iterable

from .schema import Chunk, Document

TOKEN_RE = re.compile(r"(?u)[\w][\w\-.]{1,}")
TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.I | re.S)


def stable_id(*parts: str) -> str:
    h = sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", "ignore"))
        h.update(b"\0")
    return h.hexdigest()


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def html_to_text(raw_html: str) -> str:
    without_scripts = SCRIPT_STYLE_RE.sub(" ", raw_html or "")
    without_tags = TAG_RE.sub(" ", without_scripts)
    return compact_text(html.unescape(without_tags))


def chunk_document(doc: Document, max_tokens: int = 320, overlap: int = 64) -> list[Chunk]:
    words = compact_text(doc.text).split()
    if not words:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    overlap = max(0, min(overlap, max_tokens - 1))
    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        text = " ".join(words[start:end])
        chunks.append(
            Chunk(
                id=stable_id(doc.id, str(ordinal), text[:240]),
                doc_id=doc.id,
                source=doc.source,
                kind=doc.kind,
                title=doc.title,
                uri=doc.uri,
                text=text,
                ordinal=ordinal,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                metadata=doc.metadata,
            )
        )
        if end >= len(words):
            break
        start = end - overlap
        ordinal += 1
    return chunks


def chunk_documents(docs: Iterable[Document], max_tokens: int = 320, overlap: int = 64) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, max_tokens=max_tokens, overlap=overlap))
    return chunks
