import pytest

from omozo import db
from omozo.embed import EmbeddingClient


def test_bootstrap_first_run_records_meta(conn):
    db.bootstrap(conn, dim=8, embed_model="bge-m3")
    assert db.get_meta(conn, "embed_dim") == "8"
    assert db.get_meta(conn, "embed_model") == "bge-m3"


def test_bootstrap_same_dim_and_model_ok(conn):
    db.bootstrap(conn, dim=8, embed_model="bge-m3")
    db.bootstrap(conn, dim=8, embed_model="bge-m3")  # no raise


def test_bootstrap_rejects_dim_change(conn):
    db.bootstrap(conn, dim=8, embed_model="bge-m3")
    with pytest.raises(db.MetaMismatch, match="dimension mismatch"):
        db.bootstrap(conn, dim=16, embed_model="bge-m3")


def test_bootstrap_rejects_model_change(conn):
    db.bootstrap(conn, dim=8, embed_model="bge-m3")
    with pytest.raises(db.MetaMismatch, match="model mismatch"):
        db.bootstrap(conn, dim=8, embed_model="other-model")


def test_open_existing_returns_dim(conn):
    db.bootstrap(conn, dim=8, embed_model="bge-m3")
    assert db.open_existing(conn) == 8


def test_connect_enables_wal(tmp_path):
    conn = db.connect(tmp_path / "wal.db")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_open_existing_fails_on_fresh_db(tmp_path):
    fresh = db.connect(tmp_path / "fresh.db")
    with pytest.raises(db.MetaMismatch, match="not initialized"):
        db.open_existing(fresh)


def _transport(payload: dict):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def test_embed_orders_by_index():
    client = EmbeddingClient(
        "http://x",
        "m",
        transport=_transport(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        ),
    )
    assert client.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_embed_http_error_raises():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = EmbeddingClient("http://x", "m", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="embedding request failed"):
        client.embed(["a"])


def test_embed_count_mismatch_raises():
    client = EmbeddingClient(
        "http://x",
        "m",
        transport=_transport({"data": [{"index": 0, "embedding": [1.0]}]}),
    )
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 inputs"):
        client.embed(["a", "b"])


def test_embed_inconsistent_dims_raise():
    client = EmbeddingClient(
        "http://x",
        "m",
        transport=_transport(
            {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [1.0]},
                ]
            }
        ),
    )
    with pytest.raises(RuntimeError, match="inconsistent dimensions"):
        client.embed(["a", "b"])


def test_probe_dim():
    client = EmbeddingClient(
        "http://x",
        "m",
        transport=_transport({"data": [{"index": 0, "embedding": [0.1] * 4}]}),
    )
    assert client.probe_dim() == 4
