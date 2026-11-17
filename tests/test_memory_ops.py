"""gc and transcript-search tests."""

from omozo import memory
from omozo.memory import gc, search_transcripts

NOW = "2026-11-17T12:00:00+00:00"


def _save(conn, embedder, text, kind="fact", importance=1):
    return memory.save_batch(
        conn,
        embedder,
        [{"text": text, "kind": kind, "importance": importance}],
        transcript=None,
        threshold=0.92,
        now=NOW,
    )


def test_gc_purges_old_superseded_only(conn, embedder):
    embedder.set("dark theme", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set("light theme", [0.95, 0.312, 0, 0, 0, 0, 0, 0])
    embedder.set("old fact", [0.0, 1.0, 0, 0, 0, 0, 0, 0])
    embedder.set("new fact", [0.0, 1.0, 0, 0, 0, 0, 0, 0.01])

    old = "2026-01-01T00:00:00+00:00"
    _save(conn, embedder, "dark theme", kind="preference")
    memory.save_batch(
        conn,
        embedder,
        [{"text": "light theme", "kind": "preference", "importance": 3}],
        transcript=None,
        threshold=0.92,
        now=old,  # old superseder
    )
    _save(conn, embedder, "old fact")
    # A recent supersede: same-kind near-duplicate.
    memory.save_batch(
        conn,
        embedder,
        [{"text": "new fact", "kind": "fact", "importance": 1}],
        transcript=None,
        threshold=0.92,
        now=NOW,
    )

    removed = gc(conn, age_days=30, now=NOW)
    # "dark theme" was superseded by the OLD row -> old enough to purge.
    # "old fact" was superseded by the RECENT row -> kept.
    rows = {
        r["text"]: r["status"]
        for r in conn.execute("SELECT text, status FROM memory_rows").fetchall()
    }
    assert "dark theme" not in rows
    assert rows.get("old fact") == "superseded"
    assert removed == 1


def test_gc_never_touches_active(conn, embedder):
    _save(conn, embedder, "long lived fact")
    assert gc(conn, age_days=0, now=NOW) == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM memory_rows").fetchone()["n"] == 1


def test_transcript_search_hits_fts(conn, embedder):
    memory.save_batch(
        conn,
        embedder,
        [
            {
                "text": "Discussed lighthouse maintenance",
                "kind": "summary",
                "importance": 2,
            }
        ],
        transcript="We talked about the lighthouse keeper and the fog signal.",
        threshold=0.92,
        now=NOW,
    )
    out = search_transcripts(conn, "fog signal")
    assert "#1" in out
    assert "lighthouse" in out


def test_transcript_search_no_match(conn, embedder):
    memory.save_batch(
        conn,
        embedder,
        [{"text": "Summary", "kind": "summary", "importance": 1}],
        transcript="Nothing about submarines here.",
        threshold=0.92,
        now=NOW,
    )
    assert search_transcripts(conn, "kayak") == "No transcripts found."


def test_transcript_search_empty_query(conn, embedder):
    assert search_transcripts(conn, "  ") == "No transcripts found."
