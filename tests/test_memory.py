import pytest

from omozo import memory

# ---------------------------------------------------------------- validation


def test_validate_rejects_empty_batch():
    with pytest.raises(ValueError, match="1..20"):
        memory.validate_facts([])


def test_validate_rejects_bad_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        memory.validate_facts([{"text": "x", "kind": "vibe", "importance": 3}])


def test_validate_rejects_importance_out_of_range():
    with pytest.raises(ValueError, match="importance"):
        memory.validate_facts([{"text": "x", "kind": "fact", "importance": 6}])


def test_validate_rejects_blank_text_and_lists_all_problems():
    with pytest.raises(ValueError, match="text is empty; .*importance"):
        memory.validate_facts(
            [
                {"text": "  ", "kind": "fact", "importance": 0},
                {"text": "ok", "kind": "weird", "importance": 2},
            ]
        )


def test_validate_strips_text():
    facts = memory.validate_facts([{"text": "  hi  ", "kind": "fact", "importance": 1}])
    assert facts[0].text == "hi"


# ---------------------------------------------------------------- save_batch


def test_save_batch_persists_rows(conn, embedder):
    report = memory.save_batch(
        conn,
        embedder,
        [{"text": "User prefers dark themes", "kind": "preference", "importance": 4}],
        transcript=None,
        threshold=0.92,
    )
    assert report["saved"] == [1]
    row = conn.execute("SELECT * FROM memory_rows WHERE id = 1").fetchone()
    assert row["text"] == "User prefers dark themes"
    assert row["status"] == "active"
    assert conn.execute("SELECT rowid FROM memories WHERE rowid = 1").fetchone()


def test_save_batch_drops_within_batch_duplicates(conn, embedder):
    embedder.set("likes tea", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set("drinks tea", [0.95, 0.312, 0, 0, 0, 0, 0, 0])  # cos ~0.95
    embedder.set("unrelated fact", [0, 1.0, 0, 0, 0, 0, 0, 0])
    report = memory.save_batch(
        conn,
        embedder,
        [
            {"text": "likes tea", "kind": "fact", "importance": 1},
            {"text": "drinks tea", "kind": "fact", "importance": 1},
            {"text": "unrelated fact", "kind": "fact", "importance": 1},
        ],
        transcript=None,
        threshold=0.92,
    )
    assert report["duplicates_dropped"] == 1
    assert len(report["saved"]) == 2


def test_save_batch_supersedes_conflicting_memory(conn, embedder):
    embedder.set("prefers dark theme", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set("prefers light theme", [0.95, 0.312, 0, 0, 0, 0, 0, 0])
    embedder.set("query", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    memory.save_batch(
        conn,
        embedder,
        [{"text": "prefers dark theme", "kind": "preference", "importance": 3}],
        transcript=None,
        threshold=0.92,
    )
    report = memory.save_batch(
        conn,
        embedder,
        [{"text": "prefers light theme", "kind": "preference", "importance": 3}],
        transcript=None,
        threshold=0.92,
    )
    assert report["superseded"] == [{"id": 1, "by": 2}]
    old = conn.execute(
        "SELECT status, superseded_by FROM memory_rows WHERE id = 1"
    ).fetchone()
    assert old["status"] == "superseded" and old["superseded_by"] == 2
    # Superseded row is gone from the vector index.
    assert not conn.execute("SELECT rowid FROM memories WHERE rowid = 1").fetchone()


def test_save_batch_supersede_is_kind_scoped(conn, embedder):
    embedder.set("dark note", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set("dark note again", [0.99, 0.14, 0, 0, 0, 0, 0, 0])
    memory.save_batch(
        conn,
        embedder,
        [{"text": "dark note", "kind": "fact", "importance": 2}],
        transcript=None,
        threshold=0.92,
    )
    report = memory.save_batch(
        conn,
        embedder,
        [{"text": "dark note again", "kind": "preference", "importance": 2}],
        transcript=None,
        threshold=0.92,
    )
    assert report["superseded"] == []  # different kind, no supersede


def test_save_batch_stores_transcript_and_links_summaries(conn, embedder):
    report = memory.save_batch(
        conn,
        embedder,
        [
            {"text": "Discussed Omozo schema", "kind": "summary", "importance": 2},
            {"text": "User likes sqlite", "kind": "fact", "importance": 3},
        ],
        transcript="full conversation text here",
        threshold=0.92,
    )
    conv = conn.execute("SELECT * FROM conversations").fetchone()
    assert conv["transcript"] == "full conversation text here"
    summary = conn.execute(
        "SELECT transcript_id FROM memory_rows WHERE text = 'Discussed Omozo schema'"
    ).fetchone()
    other = conn.execute(
        "SELECT transcript_id FROM memory_rows WHERE text = 'User likes sqlite'"
    ).fetchone()
    assert summary["transcript_id"] == report["conversation_id"]
    assert other["transcript_id"] is None


# ---------------------------------------------------------------- search


def test_search_finds_by_vector(conn, embedder):
    embedder.set("prefers dark themes", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set("dark mode query", [0.99, 0.14, 0, 0, 0, 0, 0, 0])
    memory.save_batch(
        conn,
        embedder,
        [{"text": "prefers dark themes", "kind": "preference", "importance": 4}],
        transcript=None,
        threshold=0.92,
    )
    out = memory.search(conn, embedder, "dark mode query", k=5, preview_chars=600)
    assert "[m1] (preference, imp 4," in out
    assert "prefers dark themes" in out


def test_search_finds_by_fts_when_vector_is_far(conn, embedder):
    # Six memories, k=2 -> vector channel returns only top-4 candidates.
    # The FTS-matching memory has the *worst* vector distance (-1 cosine),
    # so only the FTS channel can rescue it into the fused top-2.
    embedder.set("user speaks japanese", [-1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set("unrelated x", [0.0, 1.0, 0, 0, 0, 0, 0, 0])
    embedder.set("unrelated y", [0.0, 0.0, 1.0, 0, 0, 0, 0, 0])
    embedder.set("unrelated z", [0.0, 0.0, 0, 1.0, 0, 0, 0, 0])
    embedder.set("close to query", [0.99, 0.14, 0, 0, 0, 0, 0, 0])
    embedder.set("speaks japanese", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    memory.save_batch(
        conn,
        embedder,
        [
            {"text": "user speaks japanese", "kind": "fact", "importance": 2},
            {"text": "unrelated x", "kind": "fact", "importance": 1},
            {"text": "unrelated y", "kind": "fact", "importance": 1},
            {"text": "unrelated z", "kind": "fact", "importance": 1},
            {"text": "close to query", "kind": "fact", "importance": 1},
        ],
        transcript=None,
        threshold=0.92,
    )
    out = memory.search(conn, embedder, "speaks japanese", k=2, preview_chars=600)
    # m1 is retrieved via FTS despite the vector channel missing it;
    # m5 rides the vector channel.
    assert "user speaks japanese" in out
    assert "close to query" in out


def test_search_excludes_superseded(conn, embedder):
    embedder.set("dark theme", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set("light theme", [0.95, 0.312, 0, 0, 0, 0, 0, 0])
    embedder.set("theme query", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    memory.save_batch(
        conn,
        embedder,
        [{"text": "dark theme", "kind": "preference", "importance": 3}],
        transcript=None,
        threshold=0.92,
    )
    memory.save_batch(
        conn,
        embedder,
        [{"text": "light theme", "kind": "preference", "importance": 3}],
        transcript=None,
        threshold=0.92,
    )
    out = memory.search(conn, embedder, "theme query", k=5, preview_chars=600)
    assert "dark theme" not in out
    assert "light theme" in out


def test_search_respects_k(conn, embedder):
    import math as _math

    for i in range(6):
        # Spread vectors on an arc: pairwise cosine <= cos(0.5) ~ 0.88,
        # safely below the 0.92 dedupe threshold.
        embedder.set(
            f"note {i}",
            [_math.cos(i * 0.5), _math.sin(i * 0.5), 0, 0, 0, 0, 0, 0],
        )
    memory.save_batch(
        conn,
        embedder,
        [{"text": f"note {i}", "kind": "fact", "importance": 1} for i in range(6)],
        transcript=None,
        threshold=0.92,
    )
    out = memory.search(conn, embedder, "note 0", k=2, preview_chars=600)
    assert out.count("[m") == 2


def test_search_truncates_previews(conn, embedder):
    embedder.set("long", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set("q", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    long_text = "x" * 2000
    memory.save_batch(
        conn,
        embedder,
        [{"text": long_text, "kind": "fact", "importance": 1}],
        transcript=None,
        threshold=0.92,
    )
    out = memory.search(conn, embedder, "q", k=5, preview_chars=600)
    assert len(out.split("\n")[1]) <= 601  # 600 chars + ellipsis


def test_search_empty_query(conn, embedder):
    assert (
        memory.search(conn, embedder, "  ", k=5, preview_chars=600)
        == "No memories found."
    )


def test_search_no_confident_match(conn, embedder):
    # Vector-far and lexically unrelated: fused but everything pruned.
    embedder.set("japanese lighthouse keeper", [0.0, 1.0, 0, 0, 0, 0, 0, 0])
    embedder.set("zzz qqq", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    memory.save_batch(
        conn,
        embedder,
        [{"text": "japanese lighthouse keeper", "kind": "fact", "importance": 2}],
        transcript=None,
        threshold=0.92,
    )
    out = memory.search(conn, embedder, "zzz qqq", k=5, preview_chars=600)
    assert (
        out == "No confident matches found - try different wording or a broader query."
    )


def test_search_no_results_empty_db(conn, embedder):
    embedder.set("q", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    assert (
        memory.search(conn, embedder, "q", k=5, preview_chars=600)
        == "No memories found."
    )


def test_search_relative_gap_drops_imposters(conn, embedder):
    embedder.set("alpha fact", [0.95, 0.312, 0, 0, 0, 0, 0, 0])
    embedder.set("beta fact", [0.5, 0.866, 0, 0, 0, 0, 0, 0])
    embedder.set("gamma fact", [0.5, -0.866, 0, 0, 0, 0, 0, 0])
    embedder.set("query", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    memory.save_batch(
        conn,
        embedder,
        [
            {"text": "alpha fact", "kind": "fact", "importance": 1},
            {"text": "beta fact", "kind": "fact", "importance": 1},
            {"text": "gamma fact", "kind": "fact", "importance": 1},
        ],
        transcript=None,
        threshold=0.92,
    )
    # beta/gamma sit at cos 0.5, far below gap_ratio * 0.95 -> dropped.
    out = memory.search(conn, embedder, "query", k=5, preview_chars=600)
    assert out.count("[m") == 1
    assert "alpha fact" in out


def test_search_debug_scores(conn, embedder):
    embedder.set("prefers dark themes", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set("dark mode query", [0.99, 0.14, 0, 0, 0, 0, 0, 0])
    memory.save_batch(
        conn,
        embedder,
        [{"text": "prefers dark themes", "kind": "preference", "importance": 4}],
        transcript=None,
        threshold=0.92,
    )
    out = memory.search(
        conn, embedder, "dark mode query", k=5, preview_chars=600, debug_scores=True
    )
    assert "sim 0.99" in out
