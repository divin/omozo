import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from omozo import server
from omozo.server import BearerTokenMiddleware


def _mk_server(conn, embedder, **kwargs):
    return server.build_server(conn, embedder, **kwargs)


async def test_lists_exactly_three_tools(conn, embedder):
    mcp = _mk_server(conn, embedder)
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == {
            "memory_search",
            "doc_search",
            "memory_save",
        }


async def test_memory_save_and_search_roundtrip(conn, embedder):
    mcp = _mk_server(conn, embedder)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "memory_save",
            {
                "facts": [
                    {
                        "text": "User prefers concise answers",
                        "kind": "preference",
                        "importance": 4,
                    },
                    {"text": "Uses Omozo on homelab", "kind": "fact", "importance": 2},
                ],
                "transcript": "some conversation text",
            },
        )
        text = result.content[0].text
        assert "Saved 2 memories" in text
        assert "conversation #1" in text

        result = await client.call_tool("memory_search", {"query": "concise answers"})
        assert "[m1] (preference, imp 4," in result.content[0].text


async def test_doc_search_tool(conn, embedder, tmp_path):
    from omozo import ingest

    root = tmp_path / "docs-1"
    root.mkdir()
    doc = root / "harbour.md"
    doc.write_text("The harbour sleeps under morning fog.", encoding="utf-8")

    def fake_convert(file):
        return [
            ingest.Chunk(
                text="The harbour sleeps under morning fog.",
                title="harbour",
                section="Ch. 2",
            )
        ]

    ingest.ingest(conn, embedder, [doc], [root], convert_and_chunk=fake_convert)

    mcp = _mk_server(conn, embedder)
    async with Client(mcp) as client:
        result = await client.call_tool("doc_search", {"query": "fog over the water"})
        text = result.content[0].text
        assert "docs-1 » harbour.md » Ch. 2" in text
        assert "harbour sleeps" in text

        result = await client.call_tool(
            "doc_search", {"query": "fog", "source": "docs-1 » harbour.md"}
        )
        assert "harbour sleeps" in result.content[0].text


async def test_memory_save_validation_error_is_tool_error(conn, embedder):
    mcp = _mk_server(conn, embedder)
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="kind"):
            await client.call_tool(
                "memory_save",
                {"facts": [{"text": "x", "kind": "vibe", "importance": 3}]},
            )


async def test_memory_save_batch_size_limit(conn, embedder):
    mcp = _mk_server(conn, embedder)
    facts = [
        {"text": f"fact number {i}", "kind": "fact", "importance": 1} for i in range(21)
    ]
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="1..20"):
            await client.call_tool("memory_save", {"facts": facts})


async def test_memory_save_reports_supersede(conn, embedder):
    embedder.set("dark theme", [1.0, 0.0, 0, 0, 0, 0, 0, 0])
    embedder.set("light theme", [0.95, 0.312, 0, 0, 0, 0, 0, 0])
    mcp = _mk_server(conn, embedder)
    async with Client(mcp) as client:
        await client.call_tool(
            "memory_save",
            {"facts": [{"text": "dark theme", "kind": "preference", "importance": 3}]},
        )
        result = await client.call_tool(
            "memory_save",
            {"facts": [{"text": "light theme", "kind": "preference", "importance": 3}]},
        )
        assert "Superseded m1" in result.content[0].text


async def test_empty_search(conn, embedder):
    mcp = _mk_server(conn, embedder)
    async with Client(mcp) as client:
        result = await client.call_tool("memory_search", {"query": "anything"})
        assert result.content[0].text == "No memories found."


# ------------------------------------------------------------------ auth


async def test_bearer_middleware_rejects_missing_or_wrong_token():
    sent = []

    async def inner_app(scope, receive, send):
        sent.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(msg):
        responses.append(msg)

    mw = BearerTokenMiddleware(inner_app, token="secret")

    scope = {"type": "http", "headers": []}
    responses = []
    await mw(scope, receive, send)
    assert responses[0]["status"] == 401
    assert not sent

    scope = {"type": "http", "headers": [(b"authorization", b"Bearer wrong")]}
    responses = []
    await mw(scope, receive, send)
    assert responses[0]["status"] == 401
    assert not sent

    scope = {"type": "http", "headers": [(b"authorization", b"Bearer secret")]}
    await mw(scope, receive, send)
    assert len(sent) == 1


async def test_http_app_with_token_returns_401():
    app = server.http_app(_mk_server(None, None), token="secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.get("/mcp")
        assert response.status_code == 401


async def test_http_app_without_token_does_not_require_auth():
    app = server.http_app(_mk_server(None, None), token=None)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
    ):
        response = await client.get("/mcp")
        assert response.status_code != 401
