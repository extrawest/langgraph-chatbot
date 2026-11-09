"""MessagesStore against qdrant-client local mode (same query API as the server)."""

import itertools
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client.http.models import Distance

from langgraph_chatbot.chat.repositories import messages_store as store_module
from langgraph_chatbot.chat.repositories.messages_store import MessagesStore
from langgraph_chatbot.chat.vector_models.stored_message import StoredMessage

USER = uuid.uuid4()
OTHER_USER = uuid.uuid4()
CHAT = uuid.uuid4()
OTHER_CHAT = uuid.uuid4()


@pytest.fixture
def clock(monkeypatch):
    """Strictly increasing created_at, so ordering never depends on clock resolution."""
    ticks = itertools.count(1_000)
    monkeypatch.setattr(store_module, "time", SimpleNamespace(time=lambda: next(ticks)))


async def add(store, content, *, user=USER, chat=CHAT, turn=None, role="user"):
    await store.add(user_id=user, chat_id=chat, turn_id=turn or content, role=role, content=content)


async def count(qdrant) -> int:
    return (await qdrant.count("messages")).count


async def test_ensure_collection_creates_it_once(qdrant, embeddings):
    store = MessagesStore("messages", embeddings, qdrant)

    await store.ensure_collection()
    await store.ensure_collection()  # idempotent

    params = (await qdrant.get_collection("messages")).config.params.vectors
    assert (params.size, params.distance) == (768, Distance.COSINE)


async def test_ensure_collection_skips_existing(embeddings):
    client = AsyncMock()
    client.collection_exists.return_value = True

    await MessagesStore("messages", embeddings, client).ensure_collection()

    client.create_collection.assert_not_awaited()
    client.create_payload_index.assert_not_awaited()


async def test_ensure_collection_creates_tenant_and_range_indexes(embeddings):
    client = AsyncMock()
    client.collection_exists.return_value = False

    await MessagesStore("messages", embeddings, client).ensure_collection()

    fields = {c.kwargs["field_name"]: c.kwargs["field_schema"] for c in client.create_payload_index.await_args_list}
    assert set(fields) == {"group_id", "chat_id", "created_at"}
    assert fields["group_id"].is_tenant is True
    assert fields["created_at"].range is True


async def test_recent_returns_oldest_first(messages_store, clock):
    for text in ["one", "two", "three"]:
        await add(messages_store, text)

    messages = await messages_store.recent(user_id=USER, chat_id=CHAT)

    assert messages == [StoredMessage(role="user", content=c) for c in ["one", "two", "three"]]


async def test_recent_limit_keeps_the_newest(messages_store, clock):
    for text in ["one", "two", "three", "four"]:
        await add(messages_store, text)

    messages = await messages_store.recent(user_id=USER, chat_id=CHAT, limit=2)

    assert [m.content for m in messages] == ["three", "four"]


async def test_add_is_idempotent_per_turn_and_role(messages_store, qdrant):
    await add(messages_store, "draft answer", turn="t1", role="assistant")
    await add(messages_store, "final answer", turn="t1", role="assistant")  # retried node
    await add(messages_store, "question", turn="t1", role="user")

    assert await count(qdrant) == 2
    contents = {m.content for m in await messages_store.recent(user_id=USER, chat_id=CHAT)}
    assert contents == {"final answer", "question"}


async def test_recent_is_isolated_per_user_and_chat(messages_store):
    await add(messages_store, "mine")
    await add(messages_store, "other chat", chat=OTHER_CHAT)
    await add(messages_store, "other user", user=OTHER_USER)

    assert [m.content for m in await messages_store.recent(user_id=USER, chat_id=CHAT)] == ["mine"]
    assert await messages_store.recent(user_id=OTHER_USER, chat_id=OTHER_CHAT) == []


async def test_search_never_crosses_users(messages_store):
    await add(messages_store, "my cat is called Tom")
    await add(messages_store, "my cat is called Tom", user=OTHER_USER)

    results = await messages_store.search(user_id=USER, query="my cat is called Tom", limit=10)

    assert len(results) == 1


async def test_search_finds_exact_match_first(messages_store):
    for text in ["apples", "the weather in Paris", "bananas"]:
        await add(messages_store, text)

    results = await messages_store.search(user_id=USER, query="the weather in Paris")

    assert results[0].content == "the weather in Paris"


async def test_search_across_chats_or_in_one(messages_store):
    await add(messages_store, "a", chat=CHAT)
    await add(messages_store, "b", chat=OTHER_CHAT)

    assert len(await messages_store.search(user_id=USER, query="x")) == 2
    only = await messages_store.search(user_id=USER, query="x", chat_id=OTHER_CHAT)
    assert [m.content for m in only] == ["b"]


async def test_search_limit(messages_store):
    for i in range(6):
        await add(messages_store, f"m{i}")
    assert len(await messages_store.search(user_id=USER, query="m", limit=3)) == 3


async def test_long_content_is_cut_for_embedding_but_stored_whole(qdrant):
    embeddings = AsyncMock()
    embeddings.aembed_query.return_value = [0.1] * 768
    store = MessagesStore("messages", embeddings, qdrant)
    await store.ensure_collection()
    long_text = "x" * 5000

    await add(store, long_text)
    await store.search(user_id=USER, query="q" * 3000)

    first, second = embeddings.aembed_query.await_args_list
    assert len(first.args[0]) == 2000
    assert len(second.args[0]) == 2000
    assert (await store.recent(user_id=USER, chat_id=CHAT))[0].content == long_text


async def test_delete_chat_only_removes_that_chat_of_that_user(messages_store, qdrant):
    await add(messages_store, "doomed")
    await add(messages_store, "keep: other chat", chat=OTHER_CHAT)
    await add(messages_store, "keep: other user, same chat id", user=OTHER_USER)

    await messages_store.delete_chat(user_id=USER, chat_id=CHAT)

    assert await messages_store.recent(user_id=USER, chat_id=CHAT) == []
    assert await count(qdrant) == 2


def test_tenant_filter_always_has_group_id():
    without_chat = MessagesStore._tenant_filter(USER)
    with_chat = MessagesStore._tenant_filter(USER, CHAT)

    assert [c.key for c in without_chat.must] == ["group_id"]
    assert [c.key for c in with_chat.must] == ["group_id", "chat_id"]
    assert with_chat.must[0].match.value == str(USER)
