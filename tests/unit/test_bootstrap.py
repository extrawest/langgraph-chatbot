from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from langgraph_chatbot.ai import bootstrap
from langgraph_chatbot.ai.bootstrap import (
    FIRECRAWL_TOOLS,
    TAVILY_TOOLS,
    AppRuntime,
    _load_tools,
    build_runtime,
)
from tests.helpers import FakeUserMemory, ScriptedChatModel


class FakeAdapter:
    """Async context manager like langchain's MCPAdapter."""

    instances: list["FakeAdapter"] = []

    def __init__(self, target, tool_names=(), fail: Exception | None = None):
        self.target = target
        self._tools = [SimpleNamespace(name=n) for n in tool_names]
        self._fail = fail
        self.closed = False
        FakeAdapter.instances.append(self)

    async def __aenter__(self):
        if self._fail:
            raise self._fail
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def list_tools(self):
        return self._tools


@pytest.fixture(autouse=True)
def _reset_adapters():
    FakeAdapter.instances.clear()


# -------------------------------------------------------------- _load_tools
async def test_load_tools_keeps_only_allowed_tools(monkeypatch):
    monkeypatch.setattr(
        bootstrap, "MCPAdapter", lambda target: FakeAdapter(target, ["tavily_search", "tavily_crawl"])
    )
    async with AsyncExitStack() as stack:
        tools = await _load_tools(stack, "Tavily", "https://mcp", TAVILY_TOOLS)

        assert [t.name for t in tools] == ["tavily_search"]
        assert not FakeAdapter.instances[0].closed  # the stack owns the connection
    assert FakeAdapter.instances[0].closed


async def test_load_tools_survives_an_unavailable_server(monkeypatch, caplog):
    secret_url = "https://mcp?tavilyApiKey=SECRET"
    monkeypatch.setattr(
        bootstrap,
        "MCPAdapter",
        lambda target: FakeAdapter(target, fail=ConnectionError(f"cannot reach {secret_url}")),
    )
    async with AsyncExitStack() as stack:
        tools = await _load_tools(stack, "Tavily", secret_url, TAVILY_TOOLS)

    assert tools == []
    assert "Tavily MCP unavailable (ConnectionError)" in caplog.text
    assert "SECRET" not in caplog.text  # the error message may contain the key


# ------------------------------------------------------------ build_runtime
@pytest.fixture
def patched(monkeypatch, scripted_llm):
    """Replace every external dependency of build_runtime; expose the doubles."""
    ns = SimpleNamespace(
        qdrant_kwargs=None,
        qdrant=AsyncQdrantClient(":memory:"),
        memory=FakeUserMemory(),
        llm=scripted_llm(),
        load_tools=AsyncMock(return_value=[]),
        build_graph=MagicMock(return_value="compiled-graph"),
    )

    def fake_qdrant(**kwargs):
        ns.qdrant_kwargs = kwargs
        return ns.qdrant

    ns.memory.aclose = AsyncMock()
    monkeypatch.setattr(bootstrap, "AsyncQdrantClient", fake_qdrant)
    monkeypatch.setattr(bootstrap.UserMemory, "from_settings", lambda settings: ns.memory)
    monkeypatch.setattr(bootstrap, "build_llm", lambda settings: ns.llm)
    monkeypatch.setattr(bootstrap, "_load_tools", ns.load_tools)
    monkeypatch.setattr(bootstrap, "build_graph", ns.build_graph)
    return ns


async def test_build_runtime_wires_everything(settings, patched):
    runtime = await build_runtime(settings)
    try:
        assert isinstance(runtime, AppRuntime)
        assert runtime.graph == "compiled-graph"
        assert runtime.qdrant is patched.qdrant
        assert runtime.user_memory is patched.memory
        assert patched.qdrant_kwargs == {"url": settings.vector_store_url, "api_key": None}
        assert await patched.qdrant.collection_exists("messages")
        patched.load_tools.assert_not_awaited()  # no MCP keys configured
        llm, store, memory, tools = patched.build_graph.call_args.args
        assert (llm, store, memory, tools) == (patched.llm, runtime.messages_store, patched.memory, [])
    finally:
        await runtime.aclose()


async def test_build_runtime_passes_qdrant_api_key(settings, patched):
    settings.vector_store_api_key = SecretStr("qdrant-key")
    runtime = await build_runtime(settings)
    await runtime.aclose()
    assert patched.qdrant_kwargs["api_key"] == "qdrant-key"


async def test_build_runtime_can_skip_collection_setup(settings, patched):
    runtime = await build_runtime(settings, ensure_collection=False)
    try:
        assert not await patched.qdrant.collection_exists("messages")
    finally:
        await runtime.aclose()


async def test_build_runtime_loads_mcp_tools_when_keys_are_set(settings, patched):
    settings.tavily_api_key = SecretStr("tv key&x")
    settings.firecrawl_api_key = SecretStr("fc-key")
    patched.load_tools.side_effect = [["tavily-tool"], ["firecrawl-tool"]]

    runtime = await build_runtime(settings)
    await runtime.aclose()

    tavily_call, firecrawl_call = patched.load_tools.await_args_list
    _, name, target, allowed = tavily_call.args
    assert (name, allowed) == ("Tavily", TAVILY_TOOLS)
    assert target == f"{settings.tavily_mcp_url}?tavilyApiKey=tv%20key%26x"  # url-encoded
    _, name, target, allowed = firecrawl_call.args
    assert (name, allowed) == ("Firecrawl", FIRECRAWL_TOOLS)
    assert type(target).__name__ == "Client"
    assert patched.build_graph.call_args.args[3] == ["tavily-tool", "firecrawl-tool"]


async def test_aclose_closes_memory_before_qdrant(settings, patched):
    order = []
    patched.memory.aclose.side_effect = lambda: order.append("memory")
    real_close = patched.qdrant.close

    async def close(*args, **kwargs):
        order.append("qdrant")
        await real_close()

    patched.qdrant.close = close
    runtime = await build_runtime(settings)

    await runtime.aclose()

    assert order == ["memory", "qdrant"]


async def test_failed_build_closes_what_was_already_created(settings, patched):
    patched.qdrant.close = AsyncMock()
    patched.build_graph.side_effect = RuntimeError("graph broken")

    with pytest.raises(RuntimeError, match="graph broken"):
        await build_runtime(settings)

    patched.memory.aclose.assert_awaited_once()
    patched.qdrant.close.assert_awaited_once()


def test_build_llm_uses_settings(settings):
    from langgraph_chatbot.ai.llm import build_llm

    llm = build_llm(settings)

    assert llm.model.endswith(settings.llm_model)
    assert llm.google_api_key.get_secret_value() == "test-llm-key"
    assert llm.max_retries == 1
