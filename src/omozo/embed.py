"""OpenAI-compatible embedding client."""

from __future__ import annotations

import httpx


class EmbeddingError(RuntimeError):
    """The embedding endpoint failed or returned an unusable response."""


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("embedding base URL is empty")
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._timeout = timeout
        self._client = httpx.Client(transport=transport)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts; returns vectors in input order."""
        if not texts:
            return []
        try:
            response = self._client.post(
                self._url,
                json={"model": self._model, "input": texts},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()["data"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        if len(data) != len(texts):
            raise EmbeddingError(
                f"embedding endpoint returned {len(data)} vectors for {len(texts)} inputs"
            )
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors: list[list[float]] = []
        dim: int | None = None
        for item in ordered:
            vec = item.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise EmbeddingError("embedding endpoint returned an empty vector")
            if dim is None:
                dim = len(vec)
            elif len(vec) != dim:
                raise EmbeddingError(
                    "embedding endpoint returned inconsistent dimensions"
                )
            vectors.append([float(x) for x in vec])
        return vectors

    def probe_dim(self) -> int:
        """Embed a probe string and return the dimension (startup validation)."""
        return len(self.embed(["omozo"])[0])
