# MCP tool contracts

The stable surface the agent sees. Three tools, no more (ADR-0005). Tool
descriptions are written terse — a small local model reads them verbatim on
every turn.

Endpoint: `http://<server>:9095/mcp` (streamable HTTP). Auth: optional
`Authorization: Bearer <OMOZO_MCP_TOKEN>`.

---

## `memory_search`

```yaml
description: >
  Search durable memories about the user: preferences, decisions, events,
  and summaries of past conversations. Use for personal context. For
  reference material from documents, use doc_search instead.
params:
  query:  string, required — natural-language query
  k:      integer, optional, default 5
returns: |
  [m1] (preference, imp 4, 2026-01-05)
  Prefers concise answers without preamble.

  [m2] (decision, imp 3, 2026-02-11)
  Omozo stores transcripts as blobs, not vectors.
  ...
```

- Only active (non-superseded) memories are searched.
- Results are truncated previews; ids are stable within a response.
- Hybrid: vector + full-text, rank-fused. Vector-only hits below the
  configured confidence floor are pruned; if nothing clears the floor the
  tool answers "No confident matches found — try different wording or a
  broader query." — on receiving that, rephrase or say so honestly.
  (Lexical/keyword hits are never dropped by the floor.)

## `doc_search`

```yaml
description: >
  Search the user's ingested documents: books, PDFs, markdown notes.
  Use for factual/reference lookups. For context about the user,
  use memory_search instead.
params:
  query:  string, required
  k:      integer, optional, default 5
  source: string, optional — restrict to one source; use the full
          root-prefixed label exactly as shown in results
returns: |
  [c1] docs-1 » mybook.epub » Ch. 2 — The harbour ...preview text...

  [c2] docs-2 » notes.md » Setup — ...preview text...
```

- Sources are labeled `<root> » <relative path>`; multiple ingest roots
  (read-only volume mounts) are supported and stay distinguishable even
  when file names collide.
- Metadata filter (`source`) applies inside the KNN query, not after.
- Previews truncated to `OMOZO_PREVIEW_CHARS`; raise `k` for wider recall.

## `memory_save`

```yaml
description: >
  Persist new memories. Whenever durable facts about the user have
  accumulated (a preference was stated, a decision was made, something
  notable happened), save them now as a batch of short, atomic,
  self-contained facts — without waiting to be asked. Each fact needs
  text, kind (preference|fact|decision|event|summary), importance (1-5).
  Optionally attach the conversation transcript.
params:
  facts: array of {text, kind, importance}, required, 1–20 items
  transcript: string, optional — raw conversation text; stored as a blob,
          referenced by any summary fact in this batch
errors: |
  Validation failures reject the whole batch (nothing is partially written):
  - empty/oversized text
  - kind outside the enum
  - importance outside 1–5
  - more than 20 facts per batch
```

- The agent composes facts itself (ADR-0003); the server only validates,
  deduplicates, and supers conflicting older memories (ADR-0004).
- Summary facts should state what a conversation was about and what was
  concluded, not quote it — the transcript blob holds the raw text.

---

## What is deliberately NOT here

- No `memory_get`, `doc_get`, `memory_forget`, `list_sources` — admin and
  deep-fetch operations are CLI-only (`omozo forget`, `omozo gc`,
  `omozo stats`), to keep the per-turn tool description cost fixed.
- No hook-based or scheduled memory extraction — memory exists only when the
  agent calls `memory_save` (ADR-0003).
