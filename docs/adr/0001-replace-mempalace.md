# Replace MemPalace with Omozo

MemPalace currently serves agent memory, but Omozo was designed from the
start to unify memories and document retrieval behind one small MCP surface,
so MemPalace is a system to retire, not a sibling to keep. Omozo goes live
when the three tools reach parity with the MemPalace workflows the agent
actually uses; anything worth carrying over can move via a one-off CLI
import, otherwise MemPalace is simply turned off.

## Consequences

- Memory vocabulary changes: MemPalace's drawers/knowledge-graph model is
  replaced by typed flat memories with a supersede history.
- There is no transition period where both serve the agent; the MCP client
  entry points at one system at a time.
