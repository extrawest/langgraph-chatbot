import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from langgraph_chatbot.ai import memory as memory_module
from langgraph_chatbot.ai.memory import MEM0_COLLECTION, UserMemory, build_mem0_config

USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
CHAT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


# ----------------------------------------------------------------- config
@pytest.mark.parametrize(
    ("url", "host", "port"),
    [
        ("http://qdrant.test:7000", "qdrant.test", 7000),
        ("http://qdrant.test", "qdrant.test", 6333),
        ("not-a-url", "localhost", 6333),
    ],
)
def test_mem0_config_local_qdrant_uses_host_and_port(settings, url, host, port):
    settings.vector_store_url = url

    vector = build_mem0_config(settings)["vector_store"]

    assert vector["provider"] == "qdrant"
    assert vector["config"] == {
        "collection_name": MEM0_COLLECTION,
        "embedding_model_dims": settings.embedding_dims,
        "host": host,
        "port": port,
    }


def test_mem0_config_cloud_qdrant_uses_url_and_key(settings):
    settings.vector_store_url = "https://cluster.cloud.qdrant.io"
    settings.vector_store_api_key = SecretStr("cloud-key")

    cfg = build_mem0_config(settings)["vector_store"]["config"]

    assert cfg["url"] == "https://cluster.cloud.qdrant.io"
    assert cfg["api_key"] == "cloud-key"
    assert "host" not in cfg and "port" not in cfg


def test_mem0_config_llm_and_embedder(settings):
    cfg = build_mem0_config(settings)

    assert cfg["llm"] == {
        "provider": "openai",
        "config": {
            "model": settings.llm_model,
            "api_key": "test-llm-key",
            "openai_base_url": settings.llm_base_url,
        },
    }
    assert cfg["embedder"]["config"]["model"] == settings.embedding_model
    assert cfg["embedder"]["config"]["embedding_dims"] == settings.embedding_dims
    assert cfg["embedder"]["config"]["api_key"] == "test-llm-key"


def test_from_settings_builds_mem0_from_config(settings, monkeypatch):
    from_config = MagicMock(return_value="mem0-instance")
    monkeypatch.setattr(memory_module.AsyncMemory, "from_config", from_config)

    user_memory = UserMemory.from_settings(settings)

    from_config.assert_called_once_with(build_mem0_config(settings))
    assert user_memory._memory == "mem0-instance"


# ------------------------------------------------------------- UserMemory
@pytest.fixture
def mem0():
    return AsyncMock()


async def test_recall_scopes_search_to_user(mem0):
    mem0.search.return_value = {"results": [{"memory": "likes tea"}, {"memory": "lives in Kyiv"}]}

    result = await UserMemory(mem0).recall(user_id=USER_ID, query="drinks?", limit=3)

    assert result == ["likes tea", "lives in Kyiv"]
    mem0.search.assert_awaited_once_with(
        "drinks?", filters={"user_id": str(USER_ID)}, top_k=3
    )


async def test_recall_without_results(mem0):
    mem0.search.return_value = {}
    assert await UserMemory(mem0).recall(user_id=USER_ID, query="q") == []


async def test_remember_sends_the_exchange(mem0):
    await UserMemory(mem0).remember(
        user_id=USER_ID, chat_id=CHAT_ID, user_message="hi", assistant_message="hello"
    )

    mem0.add.assert_awaited_once_with(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        user_id=str(USER_ID),
        metadata={"chat_id": str(CHAT_ID)},
    )


async def test_remember_in_background_does_not_block_and_aclose_waits(mem0):
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_add(*args, **kwargs):
        started.set()
        await release.wait()

    mem0.add.side_effect = slow_add
    user_memory = UserMemory(mem0)

    user_memory.remember_in_background(
        user_id=USER_ID, chat_id=CHAT_ID, user_message="a", assistant_message="b"
    )
    await started.wait()
    assert len(user_memory._tasks) == 1  # strong reference kept while running

    release.set()
    await user_memory.aclose()

    mem0.add.assert_awaited_once()
    assert user_memory._tasks == set()  # discarded when done


async def test_background_failure_is_logged_not_raised(mem0, caplog):
    mem0.add.side_effect = RuntimeError("mem0 down")
    user_memory = UserMemory(mem0)

    user_memory.remember_in_background(
        user_id=USER_ID, chat_id=CHAT_ID, user_message="a", assistant_message="b"
    )
    await user_memory.aclose()

    assert "Mem0 add failed" in caplog.text
    assert "mem0 down" in caplog.text


async def test_aclose_without_tasks(mem0):
    await UserMemory(mem0).aclose()
