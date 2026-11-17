# Architecture

## System shape

```
                        home server
┌──────────────────────────────────────────────────────────┐
│  llama-server (chat)        llama-server --embedding     │
│                             llama-server --reranking     │
│        ▲   bge-m3 / bge-reranker-v2-m3   ▲               │
│        │                                 │               │
│  ┌─────┴──────────────────────────────────┴─────┐         │
│  │  omozo (Docker container, port 9095)         │         │
│  │  FastMCP · streamable HTTP at /mcp           │         │
│  │  optional bearer token                       │         │
│  │                                              │         │
│  │  omozo.db                                    │         │
│  │   ├─ memories      (vec0 + metadata cols)    │         │
│  │   ├─ memories_fts  (FTS5)                    │         │
│  │   ├─ chunks        (vec0 + metadata cols)    │         │
│  │   ├─ chunks_fts    (FTS5)                    │         │
│  │   ├─ conversations (transcript blobs)        │         │
│  │   ├─ sources       (path, hash, mtime)       │         │
│  │   └─ meta          (embed model, dim)        │         │
│  └──────────────────────────────────────────────┘         │
│        ▲                                       ▲          │
│        │ MCP (HTTP)                            │ exec     │
│  ┌─────┴────────────┐              ┌────────────┴───────┐  │
│  │ local agent      │              │ omozo ingest / gc  │  │
│  │ (laptop)         │              │ docker compose exec│  │
│  └──────────────────┘              └────────────────────┘  │
│                                             reads         │
│                                   /data/docs (read-only)   │
└──────────────────────────────────────────────────────────┘
```

- **MCP layer** — FastMCP, streamable HTTP transport, exactly three tools
  (see [mcp-tools.md](mcp-tools.md)). Admin operations are CLI-only.
- **Storage** — one SQLite file. `vec0` virtual tables carry metadata columns
  so `status`, `kind`, and `source` filter *inside* KNN queries. FTS5 tables
  are maintained alongside for lexical recall.
- **Embeddings** — remote llama-server (`--embedding`), model and URL from
  env, bge-m3 default (1024-dim). At startup Omozo embeds a probe string and
  refuses to run if the returned dimension differs from `meta`.
- **Rerank** — remote llama-server (`--reranking`), `bge-reranker-v2-m3`,
  doc_search only, behind a flag, degrades to no-rerank on failure.
- **Ingestion** — batch CLI (hand-invoked or via the `ingest-worker` compose
  sidecar), Docling converts (PDF/EPUB/MD/…), hybrid
  chunker aligned to the embedding tokenizer, content-hash skip per file.

## Retrieval pipeline (both searches)

1. Candidate fetch: vec0 KNN top `2k` (metadata-filtered) **and** FTS5 top
   `2k` (same filters).
2. Pre-fusion confidence prune: vector candidates below `*_MIN_SIM` are
   dropped here — dead candidates must not consume KNN slots or distort RRF
   ranks. If every vector candidate was pruned and FTS found nothing, the
   search returns "No confident matches found" (a rephrase hint, distinct
   from an empty index).
3. Reciprocal rank fusion (`k = 60`) over the two ranked lists.
4. Confidence filter: absolute floor (FTS-matched hits exempt) + relative
   gap (drop hits below `OMOZO_RELATIVE_GAP` × best hit, FTS exempt).
5. doc_search only: cross-encoder rerank of the top 20 fused candidates
   (Phase 3, config flag).
6. Truncate to `k`, format compactly with stable ids and truncated previews
   (`OMOZO_PREVIEW_CHARS`). `OMOZO_DEBUG_SCORES=1` appends similarity scores
   to each hit — used for calibrating the `*_MIN_SIM` thresholds.

## Memory write pipeline (`memory_save`)

1. Validate the fact batch: text non-empty, `kind` in
   `preference|fact|decision|event|summary`, importance `1–5`.
2. For each fact, KNN against active memories of the **same kind**:
   cosine similarity ≥ `OMOZO_DEDUPE_THRESHOLD` (default 0.92) marks the old
   row as superseded (`status = superseded`, `superseded_by` set). Newer row
   wins; nothing is deleted.
3. Insert facts; if a transcript is attached, insert the blob and reference
   it from the batch's summary fact.
4. Update `last_accessed` on any memory rows the dedupe pass touched.

## Ingestion pipeline (`omozo ingest <path>…`)

1. One or more roots can be given on the command line; `--all` walks every
   root from `OMOZO_DOCS_DIRS` (colon-separated env, matching the container's
   read-only volume mounts, e.g. `/data/docs-1:/data/docs-2`).
2. Walk the tree; for each file, hash content and compare with `sources`.
3. Unchanged → skip. Changed or new → Docling convert, hybrid chunk
   (tokenizer = embedding model), contextualize with section headers.
4. Embed chunks, replace all rows for that source (delete + insert), update
   the FTS table and `sources` row.
5. `omozo ingest --prune` additionally removes chunks whose source file no
   longer exists.

Multiple ingest roots are supported: each root gets a short label (its
mount name, e.g. `docs-1`), which prefixes the source shown to the agent
(`docs-1 » books/notes.md`) so identically-named files in different roots
stay distinguishable in `doc_search` results and its `source` filter.

## Failure behavior

- Embed endpoint unreachable / dim mismatch → server aborts at startup.
- Rerank endpoint down → doc_search returns fused results, logs once.
- Transient embed failure during save → the batch is rejected, not partially
  written; the agent can retry.

## Deliberate limits

- Single user; no `user_id` anywhere (easy to add as metadata columns later).
- Transcripts are not vectorized — FTS-only, so recall of raw conversations
  is lexical by design.
- No automatic memory extraction, decay, or summarization daemons. Memory
  exists only because the agent called `memory_save`; pruning happens via
  `omozo gc`.
