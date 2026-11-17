# Single SQLite file with vec0 + FTS5

All state lives in one SQLite file (`omozo.db`) using `vec0` virtual tables
for KNN and built-in FTS5 for lexical search, fused in application code.
This was chosen over LanceDB and server-style vector stores because Omozo is
a homelab single-user service where one file means one backup, one WAL, and
zero extra infrastructure; the corpus (hundreds of thousands of chunks at
most) is well within sqlite-vec's range.

## Considered Options

- LanceDB: embedded and capable, but a second storage format and engine to
  back up and reason about, for no gain at this scale.
- sqliteai/sqlite-rag (FTS5 + vectors + RRF out of the box): attractive, but
  the fusion and filtering logic is small enough to own explicitly, and we
  keep control over dedupe/supersede behavior.

## Consequences

- Embedding dimension is baked into the `vec0` schema; changing embedding
  models requires a full re-embed, which startup validation enforces.
