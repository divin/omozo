# Glossary

- **Memory** — a durable, self-contained statement the agent chose to persist:
  a fact, preference, decision, event, or conversation summary. Not a chunk of
  a document.

- **Fact** — a neutral atomic statement about the user or their world.
- **Preference** — a durable liking, dislike, or standing instruction.
- **Decision** — a concluded choice with a rationale, e.g. "Omozo stores
  transcripts as blobs".
- **Event** — something that happened at a point in time (milestone, incident,
  completed work).
- **Summary** — a short episodic recap of one conversation, written by the
  agent; the conversation itself may be attached as a **transcript**.

- **Transcript** — the raw text of a conversation, stored as a blob and never
  vectorized. Audits a summary; searchable by full-text only.

- **Chunk** — a tokenization-aware piece of an ingested document, produced by
  Docling's hybrid chunker and contextualized with its section headers.

- **Source** — one ingested file, tracked by content hash so unchanged files
  skip re-ingestion. Shown to the agent as `<root> » <relative path>`, e.g.
  `docs-1 » books/notes.md`.

- **Ingest root** — one mounted folder the ingestion walks, e.g.
  `/data/docs-1`. Several roots can be mounted read-only side by side;
  `OMOZO_DOCS_DIRS` lists them and `omozo ingest --all` walks each.

- **Ingest loop** — the scheduled batch runner (`omozo ingest-loop --interval
  N`, compose sidecar) that triggers `omozo ingest --all` on a clock. A clock
  around a batch, not a file watcher.

- **Supersede** — the only way memories change: a new memory marks an older,
  conflicting or duplicate memory as superseded. Superseded memories stay in
  the database but are excluded from search. Nothing is deleted at save time.

- **Hybrid search** — fusing vector KNN results with FTS5 full-text results
  via reciprocal rank fusion.

- **Rerank** — an optional cross-encoder pass over fused search candidates,
  used for document search only. If the rerank endpoint fails, reranking is
  disabled until restart and searches keep the fused order.

- **GC** — the deliberate purge step (`omozo gc`): removes memories that
  have been superseded longer than an age floor. The only way rows leave
  the database.

- **Transcript search** — full-text (FTS) search over stored conversation
  transcripts, CLI-only (`omozo transcripts`). Never an MCP tool.
