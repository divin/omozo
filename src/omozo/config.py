"""Environment-driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path
    docs_dirs: tuple[Path, ...]
    embed_url: str
    embed_model: str
    rerank_url: str | None
    rerank_model: str
    mcp_token: str | None
    preview_chars: int
    dedupe_threshold: float
    memory_min_sim: float
    doc_min_sim: float
    relative_gap: float
    debug_scores: bool
    host: str
    port: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "omozo.db"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        e = os.environ if env is None else env

        def get(name: str, default: str = "") -> str:
            return e.get(name, default).strip()

        def get_int(name: str, default: int) -> int:
            raw = get(name)
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

        def get_bool(name: str, default: bool) -> bool:
            raw = get(name)
            if not raw:
                return default
            return raw.lower() in ("1", "true", "yes", "on")

        def get_float(name: str, default: float) -> float:
            raw = get(name)
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number, got {raw!r}") from exc

        preview_chars = get_int("OMOZO_PREVIEW_CHARS", 600)
        if preview_chars < 1:
            raise ValueError("OMOZO_PREVIEW_CHARS must be >= 1")

        dedupe_threshold = get_float("OMOZO_DEDUPE_THRESHOLD", 0.92)
        if not 0.0 < dedupe_threshold < 1.0:
            raise ValueError("OMOZO_DEDUPE_THRESHOLD must be in (0, 1)")

        memory_min_sim = get_float("OMOZO_MEMORY_MIN_SIM", 0.35)
        doc_min_sim = get_float("OMOZO_DOC_MIN_SIM", 0.35)
        for name, value in (
            ("OMOZO_MEMORY_MIN_SIM", memory_min_sim),
            ("OMOZO_DOC_MIN_SIM", doc_min_sim),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")

        relative_gap = get_float("OMOZO_RELATIVE_GAP", 0.75)
        if not 0.0 <= relative_gap <= 1.0:
            raise ValueError("OMOZO_RELATIVE_GAP must be in [0, 1] (0 disables)")

        docs_dirs_raw = get("OMOZO_DOCS_DIRS") or get("DOCS_DIR")
        docs_dirs = tuple(
            Path(part) for part in docs_dirs_raw.split(":") if part.strip()
        )
        return cls(
            data_dir=Path(get("DATA_DIR", "./data")),
            docs_dirs=docs_dirs,
            embed_url=get("OMOZO_EMBED_URL"),
            embed_model=get("OMOZO_EMBED_MODEL", "bge-m3"),
            rerank_url=get("OMOZO_RERANK_URL") or None,
            rerank_model=get("OMOZO_RERANK_MODEL", "bge-reranker-v2-m3"),
            mcp_token=get("OMOZO_MCP_TOKEN") or None,
            preview_chars=preview_chars,
            dedupe_threshold=dedupe_threshold,
            memory_min_sim=memory_min_sim,
            doc_min_sim=doc_min_sim,
            relative_gap=relative_gap,
            debug_scores=get_bool("OMOZO_DEBUG_SCORES", False),
            host=get("OMOZO_HOST", "0.0.0.0"),
            port=get_int("OMOZO_PORT", 8000),
        )
