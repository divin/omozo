"""Shared fixtures: deterministic fake embedder and test database."""

from __future__ import annotations

import hashlib
import math

import pytest

from omozo import db


class FakeEmbedder:
    """Deterministic embedder: preset vectors by text, hashed fallback."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.vectors: dict[str, list[float]] = {}

    def set(self, text: str, vec: list[float]) -> None:
        norm = math.sqrt(sum(x * x for x in vec))
        self.vectors[text] = [x / norm for x in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = self.vectors.get(text)
            if vec is None:
                digest = hashlib.sha256(text.encode()).digest()
                raw = [digest[i % len(digest)] + 1.0 for i in range(self.dim)]
                norm = math.sqrt(sum(x * x for x in raw))
                vec = [x / norm for x in raw]
            out.append(vec)
        return out


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def conn(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    db.init_schema(conn, 8)
    yield conn
    conn.close()
