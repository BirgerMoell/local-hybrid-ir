from pathlib import Path

from local_hybrid_ir.index import build_index
from local_hybrid_ir.ingest import ingest_paths
from local_hybrid_ir.search import search


def test_file_ingest_and_search(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("BM25 is sparse retrieval with exact term matching.", encoding="utf-8")
    (docs_dir / "b.md").write_text("Embeddings are dense vectors for semantic retrieval.", encoding="utf-8")
    index_dir = tmp_path / "index"

    docs = ingest_paths(index_dir, [docs_dir])
    idx = build_index(index_dir)
    hits = search(idx, "sparse term ranking BM25", k=1)

    assert len(docs) == 2
    assert hits
    assert hits[0].title == "a.md"


def test_filter_by_kind(tmp_path: Path):
    doc = tmp_path / "notes.txt"
    doc.write_text("Hybrid search combines BM25 and vectors.", encoding="utf-8")
    index_dir = tmp_path / "index"

    ingest_paths(index_dir, [doc])
    idx = build_index(index_dir)

    assert search(idx, "hybrid", kinds={"md"}) == []
    assert search(idx, "hybrid", kinds={"txt"})
