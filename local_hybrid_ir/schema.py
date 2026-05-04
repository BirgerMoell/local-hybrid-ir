from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    id: str
    source: str
    kind: str
    title: str
    uri: str
    text: str
    created_at: str
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "Document":
        return cls(
            id=str(row["id"]),
            source=str(row.get("source") or ""),
            kind=str(row.get("kind") or ""),
            title=str(row.get("title") or ""),
            uri=str(row.get("uri") or ""),
            text=str(row.get("text") or ""),
            created_at=str(row.get("created_at") or ""),
            updated_at=row.get("updated_at"),
            metadata=dict(row.get("metadata") or {}),
        )


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    kind: str
    title: str
    uri: str
    text: str
    ordinal: int
    created_at: str
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "Chunk":
        return cls(
            id=str(row["id"]),
            doc_id=str(row["doc_id"]),
            source=str(row.get("source") or ""),
            kind=str(row.get("kind") or ""),
            title=str(row.get("title") or ""),
            uri=str(row.get("uri") or ""),
            text=str(row.get("text") or ""),
            ordinal=int(row.get("ordinal") or 0),
            created_at=str(row.get("created_at") or ""),
            updated_at=row.get("updated_at"),
            metadata=dict(row.get("metadata") or {}),
        )
