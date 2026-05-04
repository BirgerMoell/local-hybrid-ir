from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .index import build_index, chunks_path, hybrid_index_path, load_index
from .ingest import documents_path, ingest_paths, ingest_urls
from .io import default_index_dir, load_chunks, load_documents
from .search import search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="local-hybrid-ir",
        description="Build and query a local BM25 + dense hybrid retrieval index.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--index-dir", type=Path, default=default_index_dir(), help="Directory for documents, chunks, and index files.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest local files/directories or public URLs.")
    ingest_p.add_argument("inputs", nargs="+", help="Files, directories, or URLs to ingest.")
    ingest_p.add_argument("--append", action="store_true", help="Merge into existing documents instead of replacing them.")
    ingest_p.add_argument("--glob", default="**/*", help="Glob used when ingesting directories.")

    build_p = sub.add_parser("build", help="Chunk documents and build the hybrid index.")
    _add_build_args(build_p)

    update_p = sub.add_parser("update", help="Ingest inputs and rebuild the index.")
    update_p.add_argument("inputs", nargs="+")
    update_p.add_argument("--append", action="store_true")
    update_p.add_argument("--glob", default="**/*")
    _add_build_args(update_p)

    search_p = sub.add_parser("search", help="Search the index.")
    search_p.add_argument("query")
    search_p.add_argument("-k", "--top-k", type=int, default=8)
    search_p.add_argument("--alpha", type=float, default=0.35, help="Dense score weight. BM25 weight is 1-alpha.")
    search_p.add_argument("--source", action="append", default=None, help="Restrict to a source such as file or url.")
    search_p.add_argument("--kind", action="append", default=None, help="Restrict to a document kind such as md, json, html.")
    search_p.add_argument("--json", action="store_true", help="Emit JSON instead of readable text.")
    _add_query_embedding_args(search_p)

    context_p = sub.add_parser("context", help="Emit agent-friendly JSON retrieval context.")
    context_p.add_argument("query")
    context_p.add_argument("-k", "--top-k", type=int, default=6)
    context_p.add_argument("--alpha", type=float, default=0.35)
    context_p.add_argument("--source", action="append", default=None)
    context_p.add_argument("--kind", action="append", default=None)
    _add_query_embedding_args(context_p)

    sub.add_parser("stats", help="Show index statistics.")

    args = parser.parse_args(argv)
    index_dir = args.index_dir.expanduser().resolve()

    if args.command == "ingest":
        docs = _ingest_inputs(index_dir, args.inputs, append=args.append, glob=args.glob)
        print(f"Wrote {len(docs)} documents to {documents_path(index_dir)}")
        return 0

    if args.command == "update":
        docs = _ingest_inputs(index_dir, args.inputs, append=args.append, glob=args.glob)
        idx = build_index(
            index_dir,
            embedding=args.embedding,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Wrote {len(docs)} documents and built {len(idx.chunks)} chunks in {hybrid_index_path(index_dir)}")
        return 0

    if args.command == "build":
        idx = build_index(
            index_dir,
            embedding=args.embedding,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Built {len(idx.chunks)} chunks in {hybrid_index_path(index_dir)}")
        return 0

    if args.command in {"search", "context"}:
        idx = load_index(index_dir)
        embedding, model, dim = _embedding_from_args(args, idx)
        hits = search(
            idx,
            args.query,
            k=args.top_k,
            alpha=args.alpha,
            embedding=embedding,
            embedding_model=model,
            embedding_dim=dim,
            sources=set(args.source) if args.source else None,
            kinds=set(args.kind) if args.kind else None,
        )
        if args.command == "context":
            print(
                json.dumps(
                    {
                        "query": args.query,
                        "index_dir": str(index_dir),
                        "index_metadata": idx.metadata,
                        "hits": [hit.to_json() for hit in hits],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.json:
            print(json.dumps([hit.to_json() for hit in hits], ensure_ascii=False, indent=2))
        else:
            _print_hits(hits)
        return 0

    if args.command == "stats":
        _print_stats(index_dir)
        return 0

    return 2


def _add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding", choices=["hashing", "sentence-transformers"], default="hashing")
    parser.add_argument("--embedding-model", default=None, help="Local or cached sentence-transformers model path/name.")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--chunk-tokens", type=int, default=320)
    parser.add_argument("--chunk-overlap", type=int, default=64)


def _add_query_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding", choices=["hashing", "sentence-transformers"], default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)


def _ingest_inputs(index_dir: Path, inputs: list[str], append: bool, glob: str):
    urls = [item for item in inputs if item.startswith(("http://", "https://"))]
    paths = [Path(item) for item in inputs if item not in urls]
    docs = []
    if paths:
        docs = ingest_paths(index_dir, paths, append=append, glob=glob)
        append = True
    if urls:
        docs = ingest_urls(index_dir, urls, append=append)
    return docs


def _embedding_from_args(args, idx):
    stored = idx.metadata.get("embedding") or ""
    embedding = args.embedding or ("sentence-transformers" if stored.startswith("sentence-transformers:") else "hashing")
    model = args.embedding_model
    if model is None and stored.startswith("sentence-transformers:"):
        model = stored.split(":", 1)[1]
    dim = args.embedding_dim or int(idx.metadata.get("embedding_dim") or 384)
    return embedding, model, dim


def _print_hits(hits) -> None:
    for rank, hit in enumerate(hits, 1):
        snippet = hit.text.replace("\n", " ")
        if len(snippet) > 520:
            snippet = snippet[:517] + "..."
        print(f"{rank}. {hit.title}")
        print(f"   score={hit.score:.3f} bm25={hit.bm25_score:.3f} dense={hit.dense_score:.3f}")
        print(f"   {hit.source}/{hit.kind} {hit.uri}")
        print(f"   {snippet}\n")


def _print_stats(index_dir: Path) -> None:
    docs = load_documents(documents_path(index_dir))
    chunks = load_chunks(chunks_path(index_dir))
    sources: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for doc in docs:
        sources[doc.source] = sources.get(doc.source, 0) + 1
        kinds[doc.kind] = kinds.get(doc.kind, 0) + 1
    print(
        json.dumps(
            {
                "index_dir": str(index_dir),
                "documents": len(docs),
                "chunks": len(chunks),
                "index_exists": hybrid_index_path(index_dir).exists(),
                "sources": sources,
                "kinds": kinds,
                "documents_path": str(documents_path(index_dir)),
                "chunks_path": str(chunks_path(index_dir)),
                "index_path": str(hybrid_index_path(index_dir)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
