"""FastMCP server: the three-tool agent surface."""

from __future__ import annotations

import sqlite3
import threading
from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field, field_validator
from starlette.middleware import Middleware

from omozo import documents, memory
from omozo.embed import EmbeddingClient


class Fact(BaseModel):
    text: str
    kind: Literal["preference", "fact", "decision", "event", "summary"]
    importance: int = Field(ge=1, le=5)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text is empty")
        return stripped


class BearerTokenMiddleware:
    """ASGI middleware: require a fixed bearer token when configured."""

    def __init__(self, app, token: str) -> None:
        self.app = app
        self._expected = f"Bearer {token}"

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode("latin-1")
            if auth != self._expected:
                if scope["type"] == "http":
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 401,
                            "headers": [(b"content-type", b"text/plain")],
                        }
                    )
                    await send({"type": "http.response.body", "body": b"Unauthorized"})
                    return
                raise PermissionError("Unauthorized")
        await self.app(scope, receive, send)


def build_server(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    *,
    preview_chars: int = 600,
    dedupe_threshold: float = 0.92,
    memory_min_sim: float = 0.35,
    doc_min_sim: float = 0.35,
    relative_gap: float = 0.75,
    debug_scores: bool = False,
    reranker=None,
) -> FastMCP:
    mcp = FastMCP("omozo")
    lock = threading.Lock()

    @mcp.tool
    def memory_search(query: str, k: int = 5) -> str:
        """Search durable memories about the user: preferences, decisions, events, and summaries of past conversations. Use for personal context. For reference material from documents, use doc_search instead."""
        with lock:
            return memory.search(conn, embedder, query, k, preview_chars)

    @mcp.tool
    def doc_search(query: str, k: int = 5, source: str | None = None) -> str:
        """Search the user's ingested documents: books, PDFs, markdown notes. Use for factual/reference lookups. For context about the user, use memory_search instead."""
        with lock:
            return documents.search(
                conn,
                embedder,
                query,
                k,
                source,
                preview_chars,
                min_sim=doc_min_sim,
                gap_ratio=relative_gap,
                debug_scores=debug_scores,
                reranker=reranker,
            )

    @mcp.tool
    def memory_save(facts: list[Fact], transcript: str | None = None) -> str:
        """Persist new memories. Whenever durable facts about the user have accumulated (a preference was stated, a decision was made, something notable happened), save them now as a batch of short, atomic, self-contained facts — without waiting to be asked. Each fact states one thing. Optionally attach the raw conversation transcript."""
        try:
            report = memory.save_batch(
                conn,
                embedder,
                [fact.model_dump() for fact in facts],
                transcript,
                dedupe_threshold,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        saved = ", ".join(f"m{memory_id}" for memory_id in report["saved"]) or "none"
        lines = [f"Saved {len(report['saved'])} memories ({saved})."]
        for pair in report["superseded"]:
            lines.append(f"Superseded m{pair['id']} (replaced by m{pair['by']}).")
        if report["duplicates_dropped"]:
            lines.append(
                f"Dropped {report['duplicates_dropped']} duplicate fact(s) in this batch."
            )
        if report["conversation_id"]:
            lines.append(
                f"Transcript stored as conversation #{report['conversation_id']}."
            )
        return " ".join(lines)

    return mcp


def http_app(server: FastMCP, token: str | None = None):
    """Streamable-HTTP ASGI app with optional bearer-token auth."""
    middleware = [Middleware(BearerTokenMiddleware, token=token)] if token else None
    return server.http_app(middleware=middleware)
