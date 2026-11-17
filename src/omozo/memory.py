"""Memory save/search: validation, dedupe/supersede, hybrid retrieval."""

from __future__ import annotations

import math
import re
import sqlite3
import struct
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import sqlite_vec

from omozo import rrf
from omozo.confidence import keep_confident

FACT_KINDS = ("preference", "fact", "decision", "event", "summary")
MAX_FACTS_PER_BATCH = 20
DEDUPE_KNN_K = 8


@dataclass(frozen=True)
class FactInput:
    text: str
    kind: str
    importance: int


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def validate_facts(facts: list[dict]) -> list[FactInput]:
    """Validate a raw fact batch; raises ValueError listing every problem."""
    if not isinstance(facts, list) or not 1 <= len(facts) <= MAX_FACTS_PER_BATCH:
        raise ValueError(
            f"facts must be a list of 1..{MAX_FACTS_PER_BATCH} items, got {len(facts)}"
        )
    problems: list[str] = []
    out: list[FactInput] = []
    for i, fact in enumerate(facts):
        if not isinstance(fact, dict):
            problems.append(f"facts[{i}]: must be an object")
            continue
        text = fact.get("text")
        text = text.strip() if isinstance(text, str) else ""
        kind = fact.get("kind")
        importance = fact.get("importance")
        if not text:
            problems.append(f"facts[{i}]: text is empty")
        if kind not in FACT_KINDS:
            problems.append(f"facts[{i}]: kind must be one of {list(FACT_KINDS)}")
        if (
            not isinstance(importance, int)
            or isinstance(importance, bool)
            or not 1 <= importance <= 5
        ):
            problems.append(f"facts[{i}]: importance must be an integer in 1..5")
        if text and kind in FACT_KINDS and isinstance(importance, int):
            out.append(FactInput(text=text, kind=kind, importance=importance))
    if problems:
        raise ValueError("; ".join(problems))
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def save_batch(
    conn: sqlite3.Connection,
    embedder,
    facts: list[dict],
    transcript: str | None,
    threshold: float,
    now: str | None = None,
) -> dict:
    """Validate, deduplicate, and persist a fact batch. All-or-nothing.

    Near-duplicates within the batch are dropped; near-duplicates against
    active memories of the same kind supersede the old rows (removed from
    the vector index, kept in memory_rows with status='superseded').
    """
    now = now or utcnow_iso()
    validated = validate_facts(facts)
    vectors = embedder.embed([fact.text for fact in validated])

    kept: list[tuple[FactInput, list[float]]] = []
    duplicates_dropped = 0
    for fact, vec in zip(validated, vectors, strict=True):
        if any(
            other.kind == fact.kind and cosine(vec, other_vec) >= threshold
            for other, other_vec in kept
        ):
            duplicates_dropped += 1
        else:
            kept.append((fact, vec))

    conversation_id: int | None = None
    if transcript and transcript.strip():
        cur = conn.execute(
            "INSERT INTO conversations(created_at, transcript) VALUES (?, ?)",
            (now, transcript.strip()),
        )
        conversation_id = cur.lastrowid

    saved: list[int] = []
    superseded: list[dict] = []
    for fact, vec in kept:
        matches = _same_kind_matches(conn, fact.kind, vec, threshold)
        transcript_id = conversation_id if fact.kind == "summary" else None
        cur = conn.execute(
            "INSERT INTO memory_rows(text, kind, importance, status, created_at,"
            " last_accessed, superseded_by, transcript_id)"
            " VALUES (?, ?, ?, 'active', ?, ?, NULL, ?)",
            (fact.text, fact.kind, fact.importance, now, now, transcript_id),
        )
        new_id = cur.lastrowid
        if new_id is None:  # pragma: no cover - sqlite always sets rowid on INSERT
            raise RuntimeError("INSERT into memory_rows did not yield a rowid")
        conn.execute(
            "INSERT INTO memories(rowid, embedding, kind, importance) VALUES (?, ?, ?, ?)",
            (new_id, _serialize(vec), fact.kind, fact.importance),
        )
        for old_id in matches:
            conn.execute(
                "UPDATE memory_rows SET status = 'superseded', superseded_by = ?,"
                " superseded_at = ?"
                " WHERE id = ?",
                (new_id, now, old_id),
            )
            conn.execute("DELETE FROM memories WHERE rowid = ?", (old_id,))
            superseded.append({"id": old_id, "by": new_id})
        saved.append(new_id)

    conn.commit()
    return {
        "saved": saved,
        "superseded": superseded,
        "duplicates_dropped": duplicates_dropped,
        "conversation_id": conversation_id,
    }


def _same_kind_matches(
    conn: sqlite3.Connection, kind: str, vec: list[float], threshold: float
) -> list[int]:
    rows = conn.execute(
        "SELECT rowid, distance FROM memories"
        " WHERE kind = :kind AND embedding MATCH :vec AND k = :k"
        " ORDER BY distance",
        {"kind": kind, "vec": _serialize(vec), "k": DEDUPE_KNN_K},
    ).fetchall()
    return [row["rowid"] for row in rows if 1.0 - row["distance"] >= threshold]


def search(
    conn: sqlite3.Connection,
    embedder,
    query: str,
    k: int,
    preview_chars: int,
    now: str | None = None,
    *,
    min_sim: float = 0.35,
    gap_ratio: float = 0.75,
    debug_scores: bool = False,
) -> str:
    """Hybrid memory search: vec0 KNN + FTS5 fused via RRF, compact output.

    Vector-only hits below min_sim (or below gap_ratio * best hit) are
    pruned; FTS-matched hits are exempt from both rules. When everything
    is pruned the response is a 'no confident matches' hint.
    """
    now = now or utcnow_iso()
    query = (query or "").strip()
    if not query:
        return "No memories found."
    k = max(1, min(int(k), 20))
    candidates = k * 2

    vec = embedder.embed([query])[0]
    knn_rows = conn.execute(
        "SELECT rowid, distance FROM memories"
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
        fts_rows = conn.execute(
            "SELECT mr.id FROM memories_fts"
            " JOIN memory_rows mr ON mr.id = memories_fts.rowid"
            " WHERE memories_fts MATCH :q AND mr.status = 'active'"
            " ORDER BY rank LIMIT :k",
            {"q": fts_query, "k": candidates},
        ).fetchall()
        fts_ranking = [row["id"] for row in fts_rows]

    # Candidates existed but none clear the confidence floor, and FTS has
    # nothing to offer either: that's a 'rephrase' situation, not an empty
    # index situation.
    if (
        not fts_ranking
        and knn_rows
        and all(1.0 - row["distance"] < min_sim for row in knn_rows)
    ):
        return "No confident matches found - try different wording or a broader query."

    fused = rrf.fuse([vector_ranking, fts_ranking])[:k]
    if not fused:
        return "No memories found."

    sim_by_id = _similarity_by_rowid(conn, "memories", vec, fused)
    kept = keep_confident(fused, sim_by_id, set(fts_ranking), min_sim, gap_ratio)
    if not kept:
        return "No confident matches found - try different wording or a broader query."

    placeholders = ",".join("?" * len(kept))
    rows = {
        row["id"]: row
        for row in conn.execute(
            f"SELECT id, text, kind, importance, created_at FROM memory_rows"
            f" WHERE id IN ({placeholders}) AND status = 'active'",
            kept,
        ).fetchall()
    }
    hits = [rows[doc_id] for doc_id in kept if doc_id in rows]
    if not hits:
        return "No memories found."

    conn.execute(
        f"UPDATE memory_rows SET last_accessed = ? WHERE id IN ({placeholders})",
        (now, *rows.keys()),
    )
    conn.commit()

    lines = []
    for hit in hits:
        text = hit["text"]
        if len(text) > preview_chars:
            text = text[:preview_chars].rstrip() + "\u2026"
        score = f", sim {sim_by_id.get(hit['id'], 0.0):.2f}" if debug_scores else ""
        lines.append(
            f"[m{hit['id']}] ({hit['kind']}, imp {hit['importance']},"
            f" {hit['created_at'][:10]}{score})\n{text}"
        )
    return "\n\n".join(lines)


def _similarity_by_rowid(
    conn: sqlite3.Connection, table: str, query_vec: list[float], ids: list[int]
) -> dict[int, float]:
    """Cosine similarity of stored vectors against the query, by rowid.

    table is an internal constant ('memories' or 'chunks'), never user input.
    """
    sims: dict[int, float] = {}
    for doc_id in ids:
        row = conn.execute(
            f"SELECT embedding FROM {table} WHERE rowid = ?", (doc_id,)
        ).fetchone()
        if row is None:
            continue
        blob = row["embedding"]
        stored = list(struct.unpack(f"{len(blob) // 4}f", blob))
        sims[doc_id] = cosine(query_vec, stored)
    return sims


def _fts_query(query: str) -> str | None:
    tokens = re.findall(r"\w+", query)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


def _serialize(vec: list[float]) -> bytes:
    """Pack a vector for sqlite-vec MATCH parameters."""
    return sqlite_vec.serialize_float32(vec)


def gc(conn: sqlite3.Connection, age_days: int, now: str | None = None) -> int:
    """Purge memories superseded more than age_days ago. Returns rows removed.

    Active memories are never touched; superseded rows are the audit trail
    of the supersede policy and only leave the database here, deliberately.
    Age is measured from superseded_at (when the row became dead), not
    created_at (a fresh memory can be superseded immediately).
    """
    now = now or utcnow_iso()
    cutoff_dt = datetime.fromisoformat(now) - timedelta(days=age_days)
    cutoff = cutoff_dt.isoformat(timespec="seconds")
    cur = conn.execute(
        "DELETE FROM memory_rows WHERE status = 'superseded' AND superseded_at < ?",
        (cutoff,),
    )
    conn.commit()
    return cur.rowcount


def search_transcripts(conn: sqlite3.Connection, query: str, limit: int = 10) -> str:
    """Full-text search over stored conversation transcripts (CLI-only)."""
    query = (query or "").strip()
    if not query:
        return "No transcripts found."
    fts_query = _fts_query(query)
    if not fts_query:
        return "No transcripts found."
    rows = conn.execute(
        "SELECT c.id, c.created_at, snippet(conversations_fts, 0, '[', ']', '\u2026', 12)"
        " AS snip FROM conversations_fts"
        " JOIN conversations c ON c.id = conversations_fts.rowid"
        " WHERE conversations_fts MATCH :q"
        " ORDER BY rank LIMIT :k",
        {"q": fts_query, "k": max(1, min(limit, 50))},
    ).fetchall()
    if not rows:
        return "No transcripts found."
    lines = [f"#{row['id']} {row['created_at'][:10]}\n{row['snip']}" for row in rows]
    return "\n\n".join(lines)
