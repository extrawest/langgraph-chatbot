"""ChatRepository + ChatService (non-streaming parts) against SQLite + in-memory Qdrant."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from langgraph_chatbot.chat.repositories.repository import ChatRepository
from langgraph_chatbot.chat.service import ChatService
from langgraph_chatbot.models.chat import Chat


@pytest.fixture
def repo(session):
    return ChatRepository(session)


@pytest.fixture
def service(session, repo, fake_graph, messages_store):
    return ChatService(session, repo, fake_graph, messages_store)


@pytest.fixture
async def alice(user_factory):
    return await user_factory(email="alice@example.com")


@pytest.fixture
async def bob(user_factory):
    return await user_factory(email="bob@example.com")


# -------------------------------------------------------------- repository
async def test_get_by_id_is_scoped_to_owner(repo, chat_factory, alice, bob):
    chat = await chat_factory(alice)

    assert (await repo.get_by_id(chat.id, alice.id)).title == "Test chat"
    assert await repo.get_by_id(chat.id, bob.id) is None
    assert await repo.get_by_id(uuid.uuid4(), alice.id) is None


async def test_list_for_user_newest_first_and_only_own(repo, chat_factory, alice, bob):
    now = datetime.now(UTC)
    old = await chat_factory(alice, "old", updated_at=now - timedelta(days=2))
    new = await chat_factory(alice, "new", updated_at=now)
    mid = await chat_factory(alice, "mid", updated_at=now - timedelta(days=1))
    await chat_factory(bob, "bob's")

    chats = await repo.list_for_user(alice.id)

    assert [c.id for c in chats] == [new.id, mid.id, old.id]


async def test_list_for_user_empty(repo, alice):
    assert await repo.list_for_user(alice.id) == []


async def test_delete_only_own_chat(repo, session, chat_factory, alice, bob):
    chat = await chat_factory(alice)

    await repo.delete(chat.id, bob.id)  # not bob's: no-op
    await session.commit()
    assert await repo.get_by_id(chat.id, alice.id) is not None

    await repo.delete(chat.id, alice.id)
    await session.commit()
    assert await repo.get_by_id(chat.id, alice.id) is None


async def test_delete_missing_chat_is_noop(repo, alice):
    await repo.delete(uuid.uuid4(), alice.id)


# ----------------------------------------------------------------- service
async def test_create_chat_persists(service, sessionmaker, alice):
    chat = await service.create_chat(alice.id, "Hello")

    async with sessionmaker() as other:
        stored = await other.get(Chat, chat.id)
    assert (stored.title, stored.user_id) == ("Hello", alice.id)
    assert stored.created_at is not None


async def test_get_or_create_chat(service, chat_factory, alice, bob):
    existing = await chat_factory(alice)

    assert (await service.get_or_create_chat(existing.id, alice.id, "ignored")).id == existing.id
    assert await service.get_or_create_chat(existing.id, bob.id, "ignored") is None
    created = await service.get_or_create_chat(None, alice.id, "Brand new")
    assert created.title == "Brand new"
    assert created.id != existing.id


async def test_rename_chat_strips_title(service, chat_factory, alice):
    chat = await chat_factory(alice, "before")

    renamed = await service.rename_chat(chat.id, alice.id, "  after  ")

    assert renamed.title == "after"
    assert (await service.get_chat(chat.id, alice.id)).title == "after"


async def test_rename_someone_elses_chat(service, chat_factory, alice, bob):
    chat = await chat_factory(alice, "mine")

    assert await service.rename_chat(chat.id, bob.id, "hacked") is None
    assert (await service.get_chat(chat.id, alice.id)).title == "mine"


async def test_delete_chat(service, chat_factory, alice):
    chat = await chat_factory(alice)

    await service.delete_chat(chat.id, alice.id)

    assert await service.get_chat(chat.id, alice.id) is None


async def test_get_messages_from_store(service, messages_store, chat_factory, alice):
    chat = await chat_factory(alice)
    await messages_store.add(user_id=alice.id, chat_id=chat.id, turn_id="t1", role="user", content="hi")
    await messages_store.add(user_id=alice.id, chat_id=chat.id, turn_id="t1", role="assistant", content="hello")

    messages = await service.get_messages(chat.id, alice.id)

    assert [(m.role, m.content) for m in messages] == [("user", "hi"), ("assistant", "hello")]


async def test_get_messages_of_someone_elses_chat(service, messages_store, chat_factory, alice, bob):
    chat = await chat_factory(alice)
    await messages_store.add(user_id=alice.id, chat_id=chat.id, turn_id="t1", role="user", content="secret")

    assert await service.get_messages(chat.id, bob.id) is None
