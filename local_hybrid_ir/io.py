from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .schema import Chunk, Document


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_index_dir() -> Path:
    return Path(os.environ.get("LOCAL_HYBRID_IR_HOME", ".ir_index")).expanduser()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count


def load_documents(path: Path) -> list[Document]:
    return [Document.from_json(row) for row in iter_jsonl(path)]


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.from_json(row) for row in iter_jsonl(path)]


def dedupe_documents(docs: Iterable[Document]) -> list[Document]:
    by_id: dict[str, Document] = {}
    for doc in docs:
        if doc.text.strip():
            by_id[doc.id] = doc
    return [by_id[key] for key in sorted(by_id)]
