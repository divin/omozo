"""CLI entry point: omozo serve / ingest / ingest-loop / stats / forget."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import uvicorn

from omozo import db, ingest, server
from omozo.config import Config
from omozo.embed import EmbeddingClient, EmbeddingError
from omozo.memory import gc, search_transcripts
from omozo.rerank import RerankClient


def _docs_dirs(cfg: Config) -> list[Path]:
    return list(cfg.docs_dirs)


def _omozo_version() -> str:
    try:
        return version("omozo")
    except PackageNotFoundError:  # pragma: no cover - dev checkout
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omozo", description="Local-first RAG and agent memory."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="run the MCP server (streamable HTTP)")

    ingest_p = sub.add_parser("ingest", help="ingest documents (by path or --all)")
    ingest_p.add_argument("paths", nargs="*", help="files or directories to ingest")
    ingest_p.add_argument(
        "--all", action="store_true", help="ingest every root from OMOZO_DOCS_DIRS"
    )
    ingest_p.add_argument(
        "--prune", action="store_true", help="also remove sources whose files are gone"
    )

    loop_p = sub.add_parser(
        "ingest-loop", help="run ingest --all on a schedule (compose sidecar)"
    )
    loop_p.add_argument(
        "--interval", type=int, default=3600, help="seconds between runs"
    )
    loop_p.add_argument(
        "--once", action="store_true", help="run once and exit (for tests)"
    )

    sub.add_parser("stats", help="show database statistics")
    forget = sub.add_parser("forget", help="delete a memory by id")
    forget.add_argument("memory_id", type=int)

    gc_p = sub.add_parser(
        "gc", help="purge superseded memories older than the age floor"
    )
    gc_p.add_argument("--age-days", type=int, default=30, help="age floor in days")

    tr_p = sub.add_parser(
        "transcripts", help="full-text search over stored conversation transcripts"
    )
    tr_p.add_argument("query")
    tr_p.add_argument("--limit", type=int, default=10)

    args = parser.parse_args(argv)
    try:
        cfg = Config.from_env()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "serve":
        return _serve(cfg)
    if args.command == "ingest":
        return _ingest(cfg, args)
    if args.command == "ingest-loop":
        return _ingest_loop(cfg, args)
    if args.command == "stats":
        return _stats(cfg)
    if args.command == "forget":
        return _forget(cfg, args.memory_id)
    if args.command == "gc":
        return _gc(cfg, args.age_days)
    if args.command == "transcripts":
        return _transcripts(cfg, args.query, args.limit)
    return 2  # pragma: no cover


def _serve(cfg: Config) -> int:
    if not cfg.embed_url:
        print("error: OMOZO_EMBED_URL is required", file=sys.stderr)
        return 2
    embedder = EmbeddingClient(cfg.embed_url, cfg.embed_model)
    try:
        dim = embedder.probe_dim()
    except EmbeddingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    conn = db.connect(cfg.db_path)
    try:
        db.bootstrap(conn, dim, cfg.embed_model)
    except db.MetaMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    app = server.http_app(
        server.build_server(
            conn,
            embedder,
            preview_chars=cfg.preview_chars,
            dedupe_threshold=cfg.dedupe_threshold,
            memory_min_sim=cfg.memory_min_sim,
            doc_min_sim=cfg.doc_min_sim,
            relative_gap=cfg.relative_gap,
            debug_scores=cfg.debug_scores,
            reranker=(
                RerankClient(cfg.rerank_url, cfg.rerank_model)
                if cfg.rerank_url
                else None
            ),
        ),
        token=cfg.mcp_token,
    )
    print(
        f"omozo {_omozo_version()} serving on http://{cfg.host}:{cfg.port}/mcp"
        f" (db: {cfg.db_path})",
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port)
    return 0


def _open_existing(cfg: Config) -> sqlite3.Connection:
    conn = db.connect(cfg.db_path)
    db.open_existing(conn)
    return conn


def _open_for_ingest(cfg: Config) -> sqlite3.Connection:
    """Open the DB, bootstrapping it (via endpoint probe) if this is first run."""
    conn = db.connect(cfg.db_path)
    try:
        db.open_existing(conn)
        return conn
    except db.MetaMismatch:
        pass
    embedder = _make_embedder(cfg)
    dim = embedder.probe_dim()
    db.bootstrap(conn, dim, cfg.embed_model)
    return conn


def _make_embedder(cfg: Config) -> EmbeddingClient:
    if not cfg.embed_url:
        print("error: OMOZO_EMBED_URL is required", file=sys.stderr)
        raise SystemExit(2)
    return EmbeddingClient(cfg.embed_url, cfg.embed_model)


def _check_model(conn: sqlite3.Connection, cfg: Config) -> None:
    stored_model = db.get_meta(conn, "embed_model")
    if stored_model != cfg.embed_model:
        print(
            f"error: embedding model mismatch: db={stored_model!r} cfg={cfg.embed_model!r}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _ingest(cfg: Config, args) -> int:
    try:
        conn = _open_for_ingest(cfg)
    except db.MetaMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _check_model(conn, cfg)
    embedder = _make_embedder(cfg)
    docs_dirs = _docs_dirs(cfg)
    if args.all:
        report = ingest.ingest_all(conn, embedder, docs_dirs, prune=args.prune)
    elif args.paths:
        report = ingest.ingest(
            conn, embedder, [Path(p) for p in args.paths], docs_dirs, prune=args.prune
        )
    else:
        print("error: give paths or --all", file=sys.stderr)
        return 2
    print(f"omozo {_omozo_version()} ingest: {report.summary()}")
    return 0


def _ingest_loop(cfg: Config, args) -> int:
    try:
        conn = _open_for_ingest(cfg)
    except db.MetaMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _check_model(conn, cfg)
    embedder = _make_embedder(cfg)
    docs_dirs = _docs_dirs(cfg)
    print(
        f"omozo {_omozo_version()} ingest-loop: interval={args.interval}s"
        f" roots={list(map(str, docs_dirs))}",
        flush=True,
    )
    while True:
        try:
            report = ingest.ingest_all(conn, embedder, docs_dirs, prune=True)
            print(f"ingest-loop: {report.summary()}", flush=True)
            for failed in report.failed:
                print(f"ingest-loop: failed: {failed}", file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive on transient failures
            print(f"ingest-loop: run failed: {exc}", file=sys.stderr, flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


def _stats(cfg: Config) -> int:
    try:
        conn = _open_existing(cfg)
    except db.MetaMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counts = {
        row["status"]: row["n"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM memory_rows GROUP BY status"
        ).fetchall()
    }
    conversations = conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()[
        "n"
    ]
    chunks = conn.execute("SELECT COUNT(*) AS n FROM chunk_rows").fetchone()["n"]
    print(f"db:            {cfg.db_path}")
    print(f"embed model:   {db.get_meta(conn, 'embed_model')}")
    print(f"embed dim:     {db.get_meta(conn, 'embed_dim')}")
    print(
        f"memories:      {counts.get('active', 0)} active, {counts.get('superseded', 0)} superseded"
    )
    print(f"conversations: {conversations}")
    print(f"chunks:        {chunks}")
    return 0


def _gc(cfg: Config, age_days: int) -> int:
    try:
        conn = _open_existing(cfg)
    except db.MetaMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    removed = gc(conn, age_days)
    print(f"gc: removed {removed} superseded memory(ies) older than {age_days} days.")
    return 0


def _transcripts(cfg: Config, query: str, limit: int) -> int:
    try:
        conn = _open_existing(cfg)
    except db.MetaMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(search_transcripts(conn, query, limit))
    return 0


def _forget(cfg: Config, memory_id: int) -> int:
    try:
        conn = _open_existing(cfg)
    except db.MetaMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cur = conn.execute("DELETE FROM memory_rows WHERE id = ?", (memory_id,))
    conn.execute("DELETE FROM memories WHERE rowid = ?", (memory_id,))
    conn.commit()
    if cur.rowcount:
        print(f"Deleted m{memory_id}.")
        return 0
    print(f"No memory with id {memory_id}.")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
