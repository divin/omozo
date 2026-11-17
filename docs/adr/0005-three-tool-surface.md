# Three-tool MCP surface, admin ops in CLI

The agent-facing MCP server exposes exactly `memory_search`, `doc_search`,
and `memory_save` — and must never grow past that. The consumer is a
context-limited local model that reads every tool description on every turn,
so each additional tool is a permanent per-turn tax, and anything not needed
at conversation time (forget, gc, stats, ingest, listing sources) belongs in
the `omozo` CLI.

## Consequences

- The agent cannot delete or manage memories; corrections happen by saving a
  superseding fact, or by a human running the CLI.
- Deep-fetch of a full chunk beyond the preview is served by re-querying
  `doc_search` (e.g. with a `source` filter), not by a dedicated get tool.
