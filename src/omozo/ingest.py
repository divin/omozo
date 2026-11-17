"""Ingestion: walk roots, hash files, convert, chunk, embed, index."""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from omozo.memory import _serialize, utcnow_iso

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".pdf", ".epub"}


@dataclass(frozen=True)
class Chunk:
    text: str
    title: str
    section: str


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"added={self.added}",
            f"updated={self.updated}",
            f"skipped={self.skipped}",
            f"chunks={self.chunks}",
        ]
        if self.removed:
            parts.append(f"removed={self.removed}")
        if self.failed:
            parts.append(f"failed={len(self.failed)}")
        return " ".join(parts)


def classify_root(path: Path, docs_dirs: list[Path]) -> tuple[Path, str]:
    """Return (root, label) for a file path: the configured root containing it.

    Paths outside any configured root fall back to their parent directory.
    """
    for root in docs_dirs:
        if path == root or root in path.parents:
            return root, root.name
    return path.parent, path.parent.name


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk(path: Path) -> list[Path]:
    """All supported files under path (recursively for directories)."""
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    files: list[Path] = []
    for item in sorted(path.rglob("*")):
        if (
            item.is_file()
            and not item.name.startswith(".")
            and item.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            files.append(item)
    return files


def ingest(
    conn: sqlite3.Connection,
    embedder,
    paths: list[Path],
    docs_dirs: list[Path],
    *,
    prune: bool = False,
    convert_and_chunk=None,
) -> IngestReport:
    """Ingest one or more paths; skip unchanged files by content hash.

    convert_and_chunk defaults to the Docling adapter (lazy import);
    tests inject a fake.
    """
    if convert_and_chunk is None:
        from omozo.docling_adapter import convert_and_chunk as _default

        convert_and_chunk = _default

    report = IngestReport()
    seen: set[Path] = set()
    for given in paths:
        if not given.exists():
            print(f"ingest: skipping missing path {given}")
            continue
        for file in _walk(given):
            seen.add(file)
            try:
                _ingest_file(conn, embedder, file, docs_dirs, convert_and_chunk, report)
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
                report.failed.append(str(file))
                print(f"ingest: FAILED {file}: {exc}", file=sys.stderr)
    if prune:
        report.removed = _prune(conn, docs_dirs, seen)
    conn.commit()
    return report


def _ingest_file(
    conn: sqlite3.Connection,
    embedder,
    file: Path,
    docs_dirs: list[Path],
    convert_and_chunk,
    report: IngestReport,
) -> bool:
    content_hash = _hash_file(file)
    row = conn.execute(
        "SELECT id, content_hash FROM sources WHERE path = ?", (str(file),)
    ).fetchone()
    if row and row["content_hash"] == content_hash:
        report.skipped += 1
        return False

    chunks = convert_and_chunk(file)
    root, label = classify_root(file, docs_dirs)
    try:
        rel_path = str(file.relative_to(root))
    except ValueError:
        rel_path = file.name

    vectors = embedder.embed([chunk.text for chunk in chunks])
    source_label = f"{label} » {rel_path}"

    source_id = _upsert_source(conn, file, root, rel_path, content_hash, len(chunks))
    _replace_chunks(conn, source_id, source_label, chunks, vectors)
    if row:
        report.updated += 1
    else:
        report.added += 1
    report.chunks += len(chunks)
    return True


def _upsert_source(
    conn: sqlite3.Connection,
    file: Path,
    root: Path,
    rel_path: str,
    content_hash: str,
    chunk_count: int,
) -> int:
    now = utcnow_iso()
    conn.execute(
        "INSERT INTO sources(path, root, rel_path, content_hash, ingested_at, chunk_count)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(path) DO UPDATE SET"
        " root = excluded.root, rel_path = excluded.rel_path,"
        " content_hash = excluded.content_hash, ingested_at = excluded.ingested_at,"
        " chunk_count = excluded.chunk_count",
        (str(file), root.name, rel_path, content_hash, now, chunk_count),
    )
    row = conn.execute("SELECT id FROM sources WHERE path = ?", (str(file),)).fetchone()
    return int(row["id"])


def _replace_chunks(
    conn: sqlite3.Connection,
    source_id: int,
    source_label: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    old_ids = [
        row["id"]
        for row in conn.execute(
            "SELECT id FROM chunk_rows WHERE source_id = ?", (source_id,)
        ).fetchall()
    ]
    for chunk_id in old_ids:
        conn.execute("DELETE FROM chunks WHERE rowid = ?", (chunk_id,))
    conn.execute("DELETE FROM chunk_rows WHERE source_id = ?", (source_id,))

    for index, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True)):
        cur = conn.execute(
            "INSERT INTO chunk_rows(source_id, chunk_index, title, section, text)"
            " VALUES (?, ?, ?, ?, ?)",
            (source_id, index, chunk.title, chunk.section, chunk.text),
        )
        chunk_id = cur.lastrowid
        if chunk_id is None:  # pragma: no cover - sqlite always sets rowid
            raise RuntimeError("INSERT into chunk_rows did not yield a rowid")
        conn.execute(
            "INSERT INTO chunks(rowid, embedding, source) VALUES (?, ?, ?)",
            (chunk_id, _serialize(vec), source_label),
        )


def _prune(conn: sqlite3.Connection, docs_dirs: list[Path], seen: set[Path]) -> int:
    """Remove sources under the configured roots whose files no longer exist."""
    removed = 0
    for row in conn.execute("SELECT id, path FROM sources").fetchall():
        path = Path(row["path"])
        root, _ = classify_root(path, docs_dirs)
        if root not in docs_dirs:
            continue
        if path in seen or path.exists():
            continue
        for chunk_row in conn.execute(
            "SELECT id FROM chunk_rows WHERE source_id = ?", (row["id"],)
        ).fetchall():
            conn.execute("DELETE FROM chunks WHERE rowid = ?", (chunk_row["id"],))
        conn.execute("DELETE FROM chunk_rows WHERE source_id = ?", (row["id"],))
        conn.execute("DELETE FROM sources WHERE id = ?", (row["id"],))
        removed += 1
    return removed


def ingest_all(
    conn: sqlite3.Connection,
    embedder,
    docs_dirs: list[Path],
    *,
    prune: bool = True,
    convert_and_chunk=None,
) -> IngestReport:
    """Ingest every configured root (what `omozo ingest --all` does)."""
    existing = [root for root in docs_dirs if root.exists()]
    missing = [root for root in docs_dirs if not root.exists()]
    if missing:
        print(f"ingest: warning, missing roots: {', '.join(map(str, missing))}")
    return ingest(
        conn,
        embedder,
        existing,
        docs_dirs,
        prune=prune,
        convert_and_chunk=convert_and_chunk,
    )
