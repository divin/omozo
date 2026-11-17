# Supersede-with-history for memory updates

When a new memory duplicates or conflicts with an active memory of the same
kind, the old row is marked `superseded` with a `superseded_by` pointer and
excluded from search — never deleted or overwritten in place. Blind append
was rejected because duplicate facts would accumulate indefinitely; overwrite
was rejected because it silently loses the prior state, which for facts about
a person's changing preferences is the valuable part.

## Consequences

- Dedupe runs at save time over the incoming batch, using same-kind cosine
  similarity against active memories (threshold configurable).
- Superseded rows accumulate and are purged only by the explicit `omozo gc`
  CLI, past an age floor.
