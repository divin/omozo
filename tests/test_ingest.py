"""Ingestion tests: walk, hash-skip, replace, prune, root labels."""

from pathlib import Path

from omozo import ingest


def fake_convert_and_chunk(file: Path):
    # One chunk per file, deterministic text derived from the file content.
    text = file.read_text(encoding="utf-8").strip()
    return [ingest.Chunk(text=text, title=file.stem, section="Chapter 1")]


def _mk_root(tmp_path: Path, name: str = "docs-1") -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_ingest_adds_sources_and_chunks(conn, embedder, tmp_path):
    root = _mk_root(tmp_path)
    _write(root, "a.md", "alpha document")
    _write(root, "sub/b.md", "beta document")

    report = ingest.ingest_all(
        conn, embedder, [root], convert_and_chunk=fake_convert_and_chunk
    )
    assert report.added == 2 and report.skipped == 0
    assert report.chunks == 2
    rows = conn.execute(
        "SELECT path, root, rel_path FROM sources ORDER BY rel_path"
    ).fetchall()
    assert rows[0]["rel_path"] == "a.md" and rows[0]["root"] == "docs-1"
    assert rows[1]["rel_path"] == "sub/b.md"


def test_ingest_second_run_skips_unchanged(conn, embedder, tmp_path):
    root = _mk_root(tmp_path)
    _write(root, "a.md", "alpha document")
    ingest.ingest_all(conn, embedder, [root], convert_and_chunk=fake_convert_and_chunk)
    report = ingest.ingest_all(
        conn, embedder, [root], convert_and_chunk=fake_convert_and_chunk
    )
    assert report.skipped == 1 and report.added == 0


def test_ingest_rechunks_changed_files(conn, embedder, tmp_path):
    root = _mk_root(tmp_path)
    _write(root, "a.md", "version one")
    ingest.ingest_all(conn, embedder, [root], convert_and_chunk=fake_convert_and_chunk)
    _write(root, "a.md", "version two with more words")
    report = ingest.ingest_all(
        conn, embedder, [root], convert_and_chunk=fake_convert_and_chunk
    )

    assert report.updated == 1 and report.added == 0
    rows = conn.execute("SELECT id FROM chunk_rows").fetchall()
    assert len(rows) == 1  # replaced, not appended
    vec_rows = conn.execute("SELECT rowid FROM chunks").fetchall()
    assert len(vec_rows) == 1
    text = conn.execute("SELECT text FROM chunk_rows").fetchone()["text"]
    assert text == "version two with more words"


def test_ingest_prune_removes_deleted_files(conn, embedder, tmp_path):
    root = _mk_root(tmp_path)
    file = _write(root, "gone.md", "doomed document")
    _write(root, "stays.md", "surviving document")
    ingest.ingest_all(conn, embedder, [root], convert_and_chunk=fake_convert_and_chunk)
    file.unlink()

    report = ingest.ingest_all(
        conn, embedder, [root], prune=True, convert_and_chunk=fake_convert_and_chunk
    )
    assert report.removed == 1
    remaining = conn.execute("SELECT rel_path FROM sources").fetchall()
    assert [r["rel_path"] for r in remaining] == ["stays.md"]
    assert conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 1


def test_ingest_ignores_unsupported_extensions(conn, embedder, tmp_path):
    root = _mk_root(tmp_path)
    _write(root, "notes.txt.orig", "not supported")
    _write(root, "image.png", "not a document")
    _write(root, "real.md", "real document")
    report = ingest.ingest_all(
        conn, embedder, [root], convert_and_chunk=fake_convert_and_chunk
    )
    assert report.added == 1


def test_ingest_multiple_roots_get_distinct_labels(conn, embedder, tmp_path):
    root1 = _mk_root(tmp_path, "docs-1")
    root2 = _mk_root(tmp_path, "docs-2")
    _write(root1, "same.md", "from root one")
    _write(root2, "same.md", "from root two")
    ingest.ingest_all(
        conn, embedder, [root1, root2], convert_and_chunk=fake_convert_and_chunk
    )
    rows = {
        r["root"]: r["rel_path"]
        for r in conn.execute("SELECT root, rel_path FROM sources").fetchall()
    }
    assert rows == {"docs-1": "same.md", "docs-2": "same.md"}


def test_classify_root_falls_back_to_parent(tmp_path):
    orphan = tmp_path / "elsewhere" / "x.md"
    orphan.parent.mkdir()
    root, label = ingest.classify_root(orphan, [tmp_path / "docs-1"])
    assert root == orphan.parent
    assert label == "elsewhere"


def test_ingest_one_bad_file_does_not_kill_the_run(conn, embedder, tmp_path, capsys):
    root = _mk_root(tmp_path)
    _write(root, "good.md", "good document")
    bad = root / "bad.md"
    bad.write_text("bad document", encoding="utf-8")

    calls = []

    def flaky_convert(file: Path):
        calls.append(file.name)
        if file.name == "bad.md":
            raise RuntimeError("conversion exploded")
        return fake_convert_and_chunk(file)

    report = ingest.ingest(
        conn, embedder, [root], [root], convert_and_chunk=flaky_convert
    )
    assert report.added == 1  # good file still ingested
    assert report.failed == [str(bad)]
    assert "FAILED" in capsys.readouterr().err
    assert calls == ["bad.md", "good.md"] or set(calls) == {"bad.md", "good.md"}


def test_ingest_skips_hidden_directories(conn, embedder, tmp_path):
    """syncthing's .stfolder/.stversions must never reach the index."""
    root = _mk_root(tmp_path)
    _write(root, "real.md", "real document")
    _write(root, ".stfolder/syncthing-marker.txt", "marker")
    _write(root, ".stversions/real~20260918.md", "old version of real.md")
    _write(root, "sub/.hidden.md", "dotfile")
    _write(root, ".secrets/api-keys.md", "should not be indexed")
    report = ingest.ingest_all(
        conn, embedder, [root], convert_and_chunk=fake_convert_and_chunk
    )
    assert report.added == 1
    paths = [r["path"] for r in conn.execute("SELECT path FROM sources").fetchall()]
    assert all(".stfolder" not in p and ".stversions" not in p for p in paths)
    assert all(not p.endswith(".hidden.md") for p in paths)


def test_ingest_lock_contention(tmp_path):
    with ingest.ingest_lock(tmp_path) as first:
        assert first is True
        with ingest.ingest_lock(tmp_path) as second:
            assert second is False  # held by this process's first holder
    # Released on exit: reacquirable.
    with ingest.ingest_lock(tmp_path) as again:
        assert again is True


def test_ingest_lock_creates_dir(tmp_path):
    lock_dir = tmp_path / "does-not-exist-yet"
    with ingest.ingest_lock(lock_dir) as acquired:
        assert acquired is True
        assert lock_dir.exists()


def test_ingest_rolls_back_failed_file_transaction(
    conn, embedder, tmp_path, monkeypatch
):
    """A file failing mid-transaction must not hold the SQLite write lock."""
    root = _mk_root(tmp_path)
    _write(root, "a.md", "alpha document")
    _write(root, "bad.md", "beta document")
    _write(root, "c.md", "gamma document")

    real = ingest._replace_chunks
    calls = {"n": 0}

    def flaky(source_id, label, chunks, vectors, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # bad.md sorts second
            raise RuntimeError("boom mid-transaction")
        return real(source_id, label, chunks, vectors, *args, **kwargs)

    monkeypatch.setattr(ingest, "_replace_chunks", flaky)
    report = ingest.ingest(
        conn, embedder, [root], [root], convert_and_chunk=fake_convert_and_chunk
    )
    assert report.failed == [str(root / "bad.md")]

    texts = {r["text"] for r in conn.execute("SELECT text FROM chunk_rows")}
    assert "alpha document" in texts and "gamma document" in texts
    assert "beta document" not in texts
    # No stuck lock: subsequent writes on the connection succeed.
    conn.execute("INSERT INTO conversations(created_at, transcript) VALUES ('x', 'y')")
    conn.commit()


def test_ingest_lock_file_name(tmp_path):
    with ingest.ingest_lock(tmp_path):
        assert (tmp_path / "omozo-ingest.lock").exists()


def test_ingest_missing_root_warns_but_survives(conn, embedder, tmp_path, capsys):
    ghost = tmp_path / "does-not-exist"
    report = ingest.ingest_all(
        conn, embedder, [ghost], convert_and_chunk=fake_convert_and_chunk
    )
    assert report.added == 0
    assert "missing roots" in capsys.readouterr().out


def test_ingest_explicit_path_outside_roots(conn, embedder, tmp_path):
    root = _mk_root(tmp_path, "docs-1")
    outside = tmp_path / "loose.md"
    outside.write_text("loose document", encoding="utf-8")
    report = ingest.ingest(
        conn, embedder, [outside], [root], convert_and_chunk=fake_convert_and_chunk
    )
    assert report.added == 1
    row = conn.execute("SELECT root, rel_path FROM sources").fetchone()
    assert row["root"] == tmp_path.name  # falls back to parent dir name
