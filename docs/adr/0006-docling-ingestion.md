# Docling for document ingestion

Document parsing uses Docling (PDF, EPUB, markdown, OCR, table structure)
with its tokenizer-aware `HybridChunker` aligned to the embedding model.
Lighter stacks (PyMuPDF4LLM + ebooklib + chonkie) were rejected because none
cover scanned PDFs and EPUB with section-aware chunking in one dependency;
Docling's heavy image and CPU cost are acceptable for an offline
`docker compose exec` job on a homelab server where nothing is latency-bound.

## Consequences

- The Docker image carries Docling's large dependency tree even though the
  MCP server path never touches it; accepted to keep a single container.
- Ingestion is batch-based, never a file watcher: work happens only when
  `omozo ingest` runs, keyed by content hash. It stays invocable by hand
  (`docker compose exec`) and may additionally be driven by a scheduled
  compose sidecar (`omozo ingest-loop --interval N`) — a clock triggering a
  batch is not a daemon embedded in the server process. Serve and ingest are
  separate processes sharing `omozo.db`, which is why WAL mode is enabled.
- No in-process scheduler in `omozo serve`: ingestion stays decoupled from
  the server's lifecycle.
