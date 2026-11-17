"""doc_search tests: hybrid retrieval over ingested chunks."""

from omozo import documents, ingest


def fake_convert_and_chunk(file):
    text = file.read_text(encoding="utf-8").strip()
    return [ingest.Chunk(text=text, title=file.stem, section="Chapter 1")]


def _populate(conn, embedder, tmp_path):
    root1 = tmp_path / "docs-1"
    root2 = tmp_path / "docs-2"
    root1.mkdir()
    root2.mkdir()
    for root, name, content in [
        (root1, "harbour.md", "The harbour sleeps under morning fog."),
        (root2, "engines.md", "Diesel engines hum in the machine room."),
        (root1, "gulls.md", "Gulls circle above the harbour masts."),
    ]:
        path = root / name
        path.write_text(content, encoding="utf-8")
    embedder.set("The harbour sleeps under morning fog.", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set(
        "Diesel engines hum in the machine room.", [0.0, 1.0, 0, 0, 0, 0, 0, 0]
    )
    embedder.set("Gulls circle above the harbour masts.", [0.9, 0.43, 0, 0, 0, 0, 0, 0])
    ingest.ingest_all(
        conn, embedder, [root1, root2], convert_and_chunk=fake_convert_and_chunk
    )


def test_search_finds_and_formats_with_root_labels(conn, embedder, tmp_path):
    _populate(conn, embedder, tmp_path)
    embedder.set("fog over the water", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    out = documents.search(
        conn, embedder, "fog over the water", k=5, source=None, preview_chars=600
    )
    assert "[c" in out
    assert "docs-1 » harbour.md » Chapter 1" in out
    assert "harbour sleeps" in out


def test_search_source_filter_restricts_results(conn, embedder, tmp_path):
    _populate(conn, embedder, tmp_path)
    embedder.set("harbour", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    out = documents.search(
        conn, embedder, "harbour", k=10, source="docs-1 » gulls.md", preview_chars=600
    )
    assert "Gulls" in out
    assert "harbour sleeps" not in out
    assert "Diesel engines" not in out


def test_search_source_filter_fts_channel(conn, embedder, tmp_path):
    _populate(conn, embedder, tmp_path)
    embedder.set("query far from everything", [0.0, 0.0, 1.0, 0, 0, 0, 0, 0])
    out = documents.search(
        conn,
        embedder,
        "machine room",
        k=5,
        source="docs-2 » engines.md",
        preview_chars=600,
    )
    assert "Diesel engines" in out


def test_search_unknown_source_returns_empty(conn, embedder, tmp_path):
    _populate(conn, embedder, tmp_path)
    out = documents.search(
        conn, embedder, "harbour", k=5, source="docs-9 » nope.md", preview_chars=600
    )
    assert out == "No documents found."


def test_search_empty_query(conn, embedder, tmp_path):
    assert documents.search(
        conn, embedder, "", k=5, source=None, preview_chars=600
    ) == ("No documents found.")


def test_search_rerank_reorders_results(conn, embedder, tmp_path):
    _populate(conn, embedder, tmp_path)
    embedder.set("harbour query", [1.0, 0.0, 0, 0, 0, 0, 0, 0])

    class PrefersGulls:
        def rerank(self, query, docs):
            return sorted(
                range(len(docs)), key=lambda i: 0 if "Gulls" in docs[i] else 1
            )

    out = documents.search(
        conn,
        embedder,
        "harbour query",
        k=3,
        source=None,
        preview_chars=600,
        reranker=PrefersGulls(),
    )
    # Without rerank the strongest vector hit (harbour.md) leads; the
    # cross-encoder prefers gulls.md.
    assert out.index("Gulls") < out.index("harbour sleeps")


def test_search_rerank_failure_degrades_to_fused_order(conn, embedder, tmp_path):
    _populate(conn, embedder, tmp_path)
    embedder.set("harbour query", [1.0, 0.0, 0, 0, 0, 0, 0, 0])

    class Broken:
        def rerank(self, query, docs):
            return None

    plain = documents.search(
        conn, embedder, "harbour query", k=3, source=None, preview_chars=600
    )
    broken = documents.search(
        conn,
        embedder,
        "harbour query",
        k=3,
        source=None,
        preview_chars=600,
        reranker=Broken(),
    )
    # A failed rerank must not change the fused ordering.
    assert broken == plain


def test_search_no_confident_match(conn, embedder, tmp_path):
    _populate(conn, embedder, tmp_path)
    # Vector-far AND lexically unrelated: fused hits all pruned.
    embedder.set("zzz qqq", [0.0, 0.0, 0, 1.0, 0, 0, 0, 0])
    out = documents.search(
        conn, embedder, "zzz qqq", k=5, source=None, preview_chars=600
    )
    assert (
        out == "No confident matches found - try different wording or a broader query."
    )


def test_search_empty_database(conn, embedder):
    embedder.set("anything", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    out = documents.search(
        conn, embedder, "anything", k=5, source=None, preview_chars=600
    )
    assert out == "No documents found."
