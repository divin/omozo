"""Confidence filtering for fused search results."""

from __future__ import annotations

from collections.abc import Sequence


def keep_confident(
    ordered_ids: Sequence[int],
    sim_by_id: dict[int, float],
    fts_matched: set[int],
    min_sim: float,
    gap_ratio: float,
) -> list[int]:
    """Filter fused hits by an absolute similarity floor and a relative gap.

    - Absolute floor: drop hits with cosine similarity < min_sim.
    - Relative gap: with gap_ratio > 0, also drop hits whose similarity is
      below gap_ratio * best similarity in the set (trims imposters trailing
      one strong hit).
    - FTS-matched ids are exempt from both rules: a lexical match is
      evidence on its own, even when the embedding disagrees.
    - Input order is preserved.
    """
    if not ordered_ids:
        return []
    best = max(sim_by_id.get(doc_id, 0.0) for doc_id in ordered_ids)
    kept: list[int] = []
    for doc_id in ordered_ids:
        sim = sim_by_id.get(doc_id, 0.0)
        if doc_id in fts_matched:
            kept.append(doc_id)
            continue
        if sim < min_sim:
            continue
        if gap_ratio > 0.0 and best > 0.0 and sim < gap_ratio * best:
            continue
        kept.append(doc_id)
    return kept
