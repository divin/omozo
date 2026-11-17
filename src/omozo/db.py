"""SQLite storage: connections, schema, metadata validation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_rows (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  kind TEXT NOT NULL,
  importance INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL,
  last_accessed TEXT NOT NULL,
  superseded_by INTEGER,
  superseded_at TEXT,
  transcript_id INTEGER
);

CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  transcript TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  root TEXT NOT NULL DEFAULT '',
  rel_path TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL,
  ingested_at TEXT NOT NULL,
  chunk_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunk_rows (
  id INTEGER PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  title TEXT NOT NULL,
  section TEXT NOT NULL,
  text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS chunk_rows_source ON chunk_rows(source_id);

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
  transcript,
  content='conversations',
  content_rowid='id',
  tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS conversations_fts_ai AFTER INSERT ON conversations BEGIN
  INSERT INTO conversations_fts(rowid, transcript) VALUES (new.id, new.transcript);
END;

CREATE TRIGGER IF NOT EXISTS conversations_fts_ad AFTER DELETE ON conversations BEGIN
  INSERT INTO conversations_fts(conversations_fts, rowid, transcript)
  VALUES ('delete', old.id, old.transcript);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS memories USING vec0(
  embedding float[{dim}] distance_metric=cosine,
  kind TEXT METADATA,
  importance INTEGER METADATA
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(
  embedding float[{dim}] distance_metric=cosine,
  source TEXT METADATA
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text,
  content='chunk_rows',
  content_rowid='id',
  tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunk_rows BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunk_rows BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  text,
  content='memory_rows',
  content_rowid='id',
  tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memory_rows BEGIN
  INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memory_rows BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF text ON memory_rows BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


class MetaMismatch(RuntimeError):
    """Stored embedding metadata disagrees with configuration."""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def get_meta(
    conn: sqlite3.Connection, key: str, default: str | None = None
) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def init_schema(conn: sqlite3.Connection, dim: int) -> None:
    conn.executescript(SCHEMA.format(dim=dim))
    conn.commit()


def bootstrap(conn: sqlite3.Connection, dim: int, embed_model: str) -> None:
    """Initialize or validate embedding metadata, then ensure the schema.

    On first run records model and dimension. On later runs refuses to run
    when either changed: a different dimension or model means the vector
    index is stale and everything must be re-embedded.
    """
    stored_model: str | None
    stored_dim: str | None
    try:
        stored_model = get_meta(conn, "embed_model")
        stored_dim = get_meta(conn, "embed_dim")
    except sqlite3.OperationalError:
        # Fresh database: the schema (and meta table) does not exist yet.
        stored_model = stored_dim = None
    if stored_model is None or stored_dim is None:
        init_schema(conn, dim)
        set_meta(conn, "embed_model", embed_model)
        set_meta(conn, "embed_dim", str(dim))
        conn.commit()
        return
    if stored_dim != str(dim):
        raise MetaMismatch(
            f"embedding dimension mismatch: database has {stored_dim}, "
            f"endpoint returned {dim}. The model changed; re-embed all data."
        )
    if stored_model != embed_model:
        raise MetaMismatch(
            f"embedding model mismatch: database was built with {stored_model!r}, "
            f"configured is {embed_model!r}. Re-embed all data or restore the model."
        )
    init_schema(conn, dim)


def open_existing(conn: sqlite3.Connection) -> int:
    """Open an already-initialized database without probing the endpoint."""
    try:
        stored_dim = get_meta(conn, "embed_dim")
    except sqlite3.OperationalError as exc:
        raise MetaMismatch("database not initialized; run `omozo serve` first") from exc
    if stored_dim is None:
        raise MetaMismatch("database not initialized; run `omozo serve` first")
    dim = int(stored_dim)
    init_schema(conn, dim)
    return dim
