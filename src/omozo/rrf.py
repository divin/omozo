"""Reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Fuse ranked id lists into a single ranking by reciprocal rank score.

    Ties break by lower id for determinism. Empty rankings are ignored.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
