# Agent-composed memories, no server-side extraction

Memories exist only because the agent calls `memory_save` with facts it
composed itself as tool arguments; Omozo never runs its own extraction
pipeline, hooks, or post-turn LLM calls. The obvious alternative — Omozo
pasting conversations into llama-server with a grammar-constrained extraction
prompt — was rejected because there is no hook point in the current
tool-loop agent, and because an explicit tool call is more predictable
behavior from a small local model than invisible magic.

## Consequences

- Omozo needs no chat-LLM endpoint at all, only embeddings (+ optional
  rerank); the config surface is smaller.
- Memory quality is bounded by the agent's discipline in composing facts;
  the tool description is the prompt, and validation rejects malformed
  batches rather than fixing them.
- Memory is incomplete by construction: conversations the agent fails to
  save leave no trace. Accepted trade-off for v1.
