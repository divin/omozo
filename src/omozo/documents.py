"""Document search (doc_search): hybrid retrieval over ingested chunks."""

from __future__ import annotations

import sqlite3

from omozo import rrf
from omozo.confidence import keep_confident

RERANK_CANDIDATES = 20
from omozo.memory import _fts_query, _serialize, _similarity_by_rowid, utcnow_iso


def search(
    conn: sqlite3.Connection,
    embedder,
    query: str,
    k: int,
    source: str | None,
    preview_chars: int,
    now: str | None = None,
    *,
    min_sim: float = 0.35,
    gap_ratio: float = 0.75,
    debug_scores: bool = False,
    reranker=None,
) -> str:
    """Hybrid chunk search: vec0 KNN + FTS5 fused via RRF, compact output."""
    now = now or utcnow_iso()
    query = (query or "").strip()
    if not query:
        return "No documents found."
    k = max(1, min(int(k), 20))
    candidates = k * 2

    vec = embedder.embed([query])[0]
    if source:
        knn_rows = conn.execute(
            "SELECT rowid, distance FROM chunks"
            " WHERE source = :source AND embedding MATCH :vec AND k = :k"
            " ORDER BY distance",
            {"source": source, "vec": _serialize(vec), "k": candidates},
        ).fetchall()
    else:
        knn_rows = conn.execute(
            "SELECT rowid, distance FROM chunks"
            " WHERE embedding MATCH :vec AND k = :k"
            " ORDER BY distance",
            {"vec": _serialize(vec), "k": candidates},
        ).fetchall()
    # Pre-fusion prune: dead candidates must not consume KNN slots or
    # distort RRF ranks.
    vector_ranking = [
        row["rowid"] for row in knn_rows if 1.0 - row["distance"] >= min_sim
    ]

    fts_query = _fts_query(query)
    fts_ranking: list[int] = []
    if fts_query:
        sql = (
            "SELECT cr.id FROM chunks_fts"
            " JOIN chunk_rows cr ON cr.id = chunks_fts.rowid"
            " JOIN sources s ON s.id = cr.source_id"
            " WHERE chunks_fts MATCH :q"
        )
        params: dict = {"q": fts_query, "k": candidates}
        if source:
            sql += " AND s.root || ' » ' || s.rel_path = :source"
            params["source"] = source
        sql += " ORDER BY rank LIMIT :k"
        fts_ranking = [row["id"] for row in conn.execute(sql, params).fetchall()]

    # Candidates existed but none clear the confidence floor, and FTS has
    # nothing to offer either: that's a 'rephrase' situation, not an empty
    # index situation.
    if (
        not fts_ranking
        and knn_rows
        and all(1.0 - row["distance"] < min_sim for row in knn_rows)
    ):
        return "No confident matches found - try different wording or a broader query."

    fused_all = rrf.fuse([vector_ranking, fts_ranking])
    if not fused_all:
        return "No documents found."

    # Rerank stage (doc_search only): reorder the top fused candidates by
    # cross-encoder score. Degradable: reranker failure returns None and we
    # keep fused order.
    candidates = fused_all[:RERANK_CANDIDATES]
    rows = _fetch_rows(conn, candidates)
    if reranker is not None and len(candidates) > 1:
        order = reranker.rerank(query, [rows[i]["text"] for i in candidates])
        if order is not None:
            candidates = [candidates[i] for i in order if i < len(candidates)]

    fused = candidates[:k]
    sim_by_id = _similarity_by_rowid(conn, "chunks", vec, fused)
    kept = keep_confident(fused, sim_by_id, set(fts_ranking), min_sim, gap_ratio)
    if not kept:
        return "No confident matches found - try different wording or a broader query."

    hits = [rows[doc_id] for doc_id in kept if doc_id in rows]
    if not hits:
        return "No documents found."

    lines = []
    for hit in hits:
        text = hit["text"]
        if len(text) > preview_chars:
            text = text[:preview_chars].rstrip() + "…"
        score = f" (sim {sim_by_id.get(hit['id'], 0.0):.2f})" if debug_scores else ""
        label = f"{hit['root']} » {hit['rel_path']}" if hit["root"] else hit["rel_path"]
        section = f" » {hit['section']}" if hit["section"] else ""
        lines.append(f"[c{hit['id']}] {label}{section}{score}\n{text}")
    return "\n\n".join(lines)


def _fetch_rows(conn, ids: list[int]) -> dict[int, sqlite3.Row]:
    placeholders = ",".join("?" * len(ids))
    return {
        row["id"]: row
        for row in conn.execute(
            f"SELECT cr.id, cr.section, cr.text, s.root, s.rel_path"
            f" FROM chunk_rows cr JOIN sources s ON s.id = cr.source_id"
            f" WHERE cr.id IN ({placeholders})",
            ids,
        ).fetchall()
    }
