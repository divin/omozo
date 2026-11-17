import httpx

from omozo.rerank import RerankClient


def _transport(payload: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def test_rerank_orders_by_relevance_score():
    client = RerankClient(
        "http://x",
        "m",
        transport=_transport(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.3},
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.5},
                ]
            }
        ),
    )
    assert client.rerank("q", ["a", "b", "c"]) == [2, 1, 0]


def test_rerank_accepts_score_field_fallback():
    client = RerankClient(
        "http://x",
        "m",
        transport=_transport(
            {"results": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}]}
        ),
    )
    assert client.rerank("q", ["a", "b"]) == [1, 0]


def test_rerank_empty_documents():
    client = RerankClient("http://x", "m", transport=_transport({"results": []}))
    assert client.rerank("q", []) == []


def test_rerank_failure_disables_until_restart():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = RerankClient("http://x", "m", transport=httpx.MockTransport(handler))
    assert client.rerank("q", ["a"]) is None  # failure -> None
    assert client.rerank("q", ["a"]) is None  # disabled, no second request


def test_rerank_malformed_results_are_skipped():
    client = RerankClient(
        "http://x",
        "m",
        transport=_transport(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": "bogus", "relevance_score": 0.5},
                    {"index": 0},
                ]
            }
        ),
    )
    assert client.rerank("q", ["a", "b"]) == [1]
