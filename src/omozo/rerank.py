"""Cross-encoder rerank client (llama-server /v1/rerank, Jina-compatible).

Degradation policy (ADR-documented): if the endpoint fails, reranking is
disabled for the remainder of the process lifetime, the failure is logged
once, and searches fall back to fused order.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("omozo.rerank")


class RerankError(RuntimeError):
    """The rerank endpoint failed or returned an unusable response."""


class RerankClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("rerank base URL is empty")
        self._url = base_url.rstrip("/") + "/rerank"
        self._model = model
        self._timeout = timeout
        self._client = httpx.Client(transport=transport)
        self._disabled = False

    def rerank(self, query: str, documents: list[str]) -> list[int] | None:
        """Return document indices best-first, or None if unavailable.

        None (instead of raising) after the first failure keeps the search
        path degradable without try/except at every call site.
        """
        if self._disabled:
            return None
        if not documents:
            return []
        try:
            response = self._client.post(
                self._url,
                json={"model": self._model, "query": query, "documents": documents},
                timeout=self._timeout,
            )
            response.raise_for_status()
            results = response.json()["results"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            self._disabled = True
            logger.warning(
                "rerank endpoint failed (%s); reranking disabled until restart", exc
            )
            return None
        scored: list[tuple[int, float]] = []
        for item in results:
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(index, int) or not isinstance(score, (int, float)):
                continue
            scored.append((index, float(score)))
        scored.sort(key=lambda pair: -pair[1])
        return [index for index, _ in scored]
