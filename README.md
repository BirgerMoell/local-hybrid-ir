# Local Hybrid IR

`local-hybrid-ir` is a small, standalone information retrieval system for local
documents. It is designed to be useful as a real command-line search tool and as a
teaching project for students learning how retrieval-augmented systems work.

The system builds a local index from files or public URLs, chunks documents into
passages, ranks results with **BM25** and a local **dense vector** score, and returns
human-readable or JSON context for downstream tools.

## Privacy And Scope

This repository is intentionally public-safe:

- No private datasets are included.
- No chat exports, private workspaces, tokens, cookies, or credentials are included.
- No organization-specific connectors are included.
- Indexed data is written to `.ir_index/`, which is ignored by git.
- The default dense embedder is deterministic and local; it does not call an API.

If you index private data locally, keep `.ir_index/` private and do not commit it.

## Why Hybrid Retrieval?

Most practical search systems combine multiple signals.

**BM25** is a sparse lexical ranker. It is excellent when a query and a document use
the same terms. For example, a query containing `BM25` will strongly match a document
that literally contains `BM25`.

**Dense retrieval** represents text as vectors. In production, this is usually done
with a trained embedding model. This project includes a dependency-free local hashing
embedder so the system works immediately and students can inspect the full pipeline.
Optional `sentence-transformers` support is available for stronger embeddings.

**Hybrid retrieval** combines these scores:

```text
final_score = alpha * dense_score + (1 - alpha) * bm25_score
```

The default `alpha` is `0.35`, giving BM25 a little more influence because the
built-in dense embedder is intentionally lightweight.

## Install

You can run the project without installing it:

```bash
./bin/local-hybrid-ir --help
```

Or install it in editable mode:

```bash
python3 -m pip install -e .
```

Optional semantic embeddings:

```bash
python3 -m pip install -e '.[semantic]'
```

## Quick Start

Build an index from the example documents:

```bash
./bin/local-hybrid-ir --index-dir .ir_index update examples
```

Search it:

```bash
./bin/local-hybrid-ir --index-dir .ir_index search "how does BM25 ranking work?"
```

Return JSON context for an agent or script:

```bash
./bin/local-hybrid-ir --index-dir .ir_index context "why combine sparse and dense retrieval?"
```

Show index statistics:

```bash
./bin/local-hybrid-ir --index-dir .ir_index stats
```

## Ingesting Your Own Data

Index a folder:

```bash
./bin/local-hybrid-ir --index-dir .ir_index update ~/notes
```

Index multiple files:

```bash
./bin/local-hybrid-ir --index-dir .ir_index update report.md faq.jsonl notes.csv
```

Index a public URL:

```bash
./bin/local-hybrid-ir --index-dir .ir_index update https://example.com
```

Append new sources to an existing index:

```bash
./bin/local-hybrid-ir --index-dir .ir_index update --append new_notes/
```

Supported local file types include:

- `.txt`, `.md`, `.rst`
- `.html`, `.htm`
- `.json`, `.jsonl`
- `.csv`, `.tsv`
- `.yaml`, `.yml`
- common code files such as `.py`, `.js`, `.ts`, `.tsx`, `.jsx`

For `.jsonl`, `.json`, `.csv`, and `.tsv`, rows with `text`, `content`, or `body`
fields are treated as separate documents. `title`, `name`, `url`, or `uri` fields
are used when present.

## Commands

### `ingest`

Reads files/directories/URLs and writes `documents.jsonl`.

```bash
./bin/local-hybrid-ir --index-dir .ir_index ingest examples
```

### `build`

Chunks `documents.jsonl`, builds BM25 and dense vectors, and writes the hybrid index.

```bash
./bin/local-hybrid-ir --index-dir .ir_index build
```

### `update`

Runs `ingest` and `build` in one step.

```bash
./bin/local-hybrid-ir --index-dir .ir_index update examples
```

### `search`

Returns ranked hits.

```bash
./bin/local-hybrid-ir --index-dir .ir_index search "retrieval evaluation" -k 5
```

Use JSON:

```bash
./bin/local-hybrid-ir --index-dir .ir_index search "retrieval evaluation" --json
```

Filter by source or kind:

```bash
./bin/local-hybrid-ir --index-dir .ir_index search "vectors" --source file --kind md
```

Tune hybrid weighting:

```bash
./bin/local-hybrid-ir --index-dir .ir_index search "exact phrase BM25" --alpha 0.25
```

Lower `alpha` gives BM25 more weight. Higher `alpha` gives dense retrieval more weight.

### `context`

Returns an agent-friendly object with query, index metadata, and hits.

```bash
./bin/local-hybrid-ir --index-dir .ir_index context "what is chunking?"
```

### `stats`

Prints document and chunk counts.

```bash
./bin/local-hybrid-ir --index-dir .ir_index stats
```

## Using A Sentence-Transformers Model

The default local hashing embedder is useful for demos and lightweight local search.
For stronger semantic retrieval, install the optional dependencies and build with a
local or cached model:

```bash
python3 -m pip install -e '.[semantic]'
./bin/local-hybrid-ir --index-dir .ir_index update examples \
  --embedding sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Query with the same embedding settings:

```bash
./bin/local-hybrid-ir --index-dir .ir_index search "semantic search" \
  --embedding sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

If the model is not already cached locally, `sentence-transformers` may download it.

## Project Structure

```text
local_hybrid_ir/
  bm25.py        # sparse BM25 ranker
  embeddings.py  # local hashing embedder and optional sentence-transformers wrapper
  ingest.py      # file, table, JSON, HTML, and public URL ingestion
  index.py       # chunking + index serialization
  search.py      # hybrid scoring and filtering
  text.py        # tokenization, HTML cleanup, chunking helpers
  cli.py         # command-line interface
examples/        # small public teaching documents
tests/           # regression tests
```

## Teaching Exercises

1. Change `--alpha` and observe how rankings change.
2. Change `--chunk-tokens` and compare short vs. long chunks.
3. Add relevance labels for a few queries and compute precision@k manually.
4. Replace the hashing embedder with a sentence-transformers model.
5. Add a reranker stage that reorders the top 20 retrieved chunks.
6. Add citations to answers generated from `context` output.

## Limitations

- The built-in hashing embedder is not a substitute for a trained semantic embedding
  model. It is included so the full system runs locally without network setup.
- URL ingestion is intentionally simple and does not crawl links recursively.
- PDF, DOCX, and other binary formats are not parsed by default.
- The index is stored as local JSONL and pickle files, which is easy to inspect but
  not meant for very large production corpora.

## Development

Run tests:

```bash
python3 -m pytest -q
```

Run a local smoke test:

```bash
./bin/local-hybrid-ir --index-dir /tmp/local-hybrid-ir-demo update examples
./bin/local-hybrid-ir --index-dir /tmp/local-hybrid-ir-demo search "hybrid retrieval"
```
