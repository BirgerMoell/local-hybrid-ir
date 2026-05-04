from __future__ import annotations

import csv
import json
import mimetypes
import urllib.request
from pathlib import Path
from typing import Iterable

from .io import dedupe_documents, load_documents, utc_now, write_jsonl
from .schema import Document
from .text import compact_text, html_to_text, stable_id


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".htm",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
}


def documents_path(index_dir: Path) -> Path:
    return index_dir / "documents.jsonl"


def ingest_paths(
    index_dir: Path,
    paths: Iterable[Path],
    append: bool = False,
    glob: str = "**/*",
) -> list[Document]:
    docs: list[Document] = []
    if append:
        docs.extend(load_documents(documents_path(index_dir)))
    for path in paths:
        docs.extend(read_path(path.expanduser(), glob=glob))
    docs = dedupe_documents(docs)
    write_jsonl(documents_path(index_dir), (doc.to_json() for doc in docs))
    return docs


def ingest_urls(index_dir: Path, urls: Iterable[str], append: bool = False) -> list[Document]:
    docs: list[Document] = []
    if append:
        docs.extend(load_documents(documents_path(index_dir)))
    for url in urls:
        docs.append(read_url(url))
    docs = dedupe_documents(docs)
    write_jsonl(documents_path(index_dir), (doc.to_json() for doc in docs))
    return docs


def read_path(path: Path, glob: str = "**/*") -> list[Document]:
    if path.is_dir():
        out: list[Document] = []
        for child in sorted(path.glob(glob)):
            if child.is_file() and child.suffix.lower() in TEXT_EXTENSIONS:
                out.extend(read_path(child))
        return out
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="replace")
    created = utc_now()
    if suffix == ".jsonl":
        return _read_jsonl(path, raw, created)
    if suffix == ".json":
        return _read_json(path, raw, created)
    if suffix in {".csv", ".tsv"}:
        return _read_table(path, raw, created, delimiter="\t" if suffix == ".tsv" else ",")
    text = html_to_text(raw) if suffix in {".html", ".htm"} else compact_text(raw)
    return [
        Document(
            id=stable_id("file", str(path.resolve())),
            source="file",
            kind=suffix.lstrip(".") or "text",
            title=path.name,
            uri=str(path.resolve()),
            text=text,
            created_at=created,
            updated_at=_mtime(path),
            metadata={"path": str(path.resolve()), "extension": suffix},
        )
    ]


def read_url(url: str) -> Document:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "local-hybrid-ir/0.1",
            "Accept": "text/html,text/plain,application/json,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        content_type = res.headers.get("Content-Type", "")
        raw = res.read().decode("utf-8", "replace")
    text = html_to_text(raw) if "html" in content_type.lower() or raw.lstrip().startswith("<") else compact_text(raw)
    title = _title_from_html(raw) or url.rstrip("/").split("/")[-1] or url
    return Document(
        id=stable_id("url", url),
        source="url",
        kind=_kind_from_content_type(content_type),
        title=title,
        uri=url,
        text=text,
        created_at=utc_now(),
        metadata={"content_type": content_type},
    )


def _read_jsonl(path: Path, raw: str, created: str) -> list[Document]:
    docs: list[Document] = []
    for i, line in enumerate(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = {"text": line}
        docs.append(_doc_from_mapping(path, row, created, suffix="jsonl", ordinal=i))
    return docs


def _read_json(path: Path, raw: str, created: str) -> list[Document]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [_plain_file_doc(path, raw, created, "json")]
    if isinstance(data, list):
        return [_doc_from_mapping(path, row if isinstance(row, dict) else {"text": row}, created, "json", i) for i, row in enumerate(data)]
    if isinstance(data, dict):
        if any(k in data for k in ("text", "content", "body")):
            return [_doc_from_mapping(path, data, created, "json", 0)]
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return [_plain_file_doc(path, text, created, "json")]
    return [_plain_file_doc(path, str(data), created, "json")]


def _read_table(path: Path, raw: str, created: str, delimiter: str) -> list[Document]:
    docs: list[Document] = []
    reader = csv.DictReader(raw.splitlines(), delimiter=delimiter)
    for i, row in enumerate(reader):
        docs.append(_doc_from_mapping(path, dict(row), created, path.suffix.lstrip("."), i))
    return docs


def _doc_from_mapping(path: Path, row: dict, created: str, suffix: str, ordinal: int) -> Document:
    text = row.get("text") or row.get("content") or row.get("body") or json.dumps(row, ensure_ascii=False, sort_keys=True)
    title = str(row.get("title") or row.get("name") or f"{path.name} row {ordinal}")
    uri = str(row.get("url") or row.get("uri") or f"{path.resolve()}#{ordinal}")
    return Document(
        id=stable_id("file-row", str(path.resolve()), str(ordinal), uri, title),
        source="file",
        kind=suffix,
        title=title,
        uri=uri,
        text=compact_text(str(text)),
        created_at=created,
        updated_at=_mtime(path),
        metadata={"path": str(path.resolve()), "ordinal": ordinal},
    )


def _plain_file_doc(path: Path, text: str, created: str, suffix: str) -> Document:
    return Document(
        id=stable_id("file", str(path.resolve())),
        source="file",
        kind=suffix,
        title=path.name,
        uri=str(path.resolve()),
        text=compact_text(text),
        created_at=created,
        updated_at=_mtime(path),
        metadata={"path": str(path.resolve())},
    )


def _mtime(path: Path) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat()


def _kind_from_content_type(content_type: str) -> str:
    if not content_type:
        return "web"
    if "html" in content_type.lower():
        return "html"
    ext = mimetypes.guess_extension(content_type.split(";", 1)[0].strip()) or ""
    return ext.lstrip(".") or "web"


def _title_from_html(raw: str) -> str | None:
    import re

    m = re.search(r"<title[^>]*>(.*?)</title>", raw or "", flags=re.I | re.S)
    return compact_text(m.group(1)) if m else None
