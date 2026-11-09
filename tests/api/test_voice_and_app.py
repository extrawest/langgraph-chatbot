from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from fastapi import FastAPI

from langgraph_chatbot import main


@pytest.fixture
async def alice(user_factory):
    return await user_factory()


def room_attrs(token: str) -> dict:
    return jwt.decode(token, options={"verify_signature": False})["attributes"]


# ------------------------------------------------------------------ voice
async def test_voice_token_creates_voice_chat(client, alice, auth_headers, settings):
    response = await client.post("/voice/token", json={}, headers=auth_headers(alice))

    assert response.status_code == 200
    body = response.json()
    assert body["url"] == settings.livekit_url
    assert room_attrs(body["token"]) == {"chat_id": body["chat_id"]}
    chats = (await client.get("/chats", headers=auth_headers(alice))).json()
    assert chats == [{"id": body["chat_id"], "title": "Voice chat"}]


async def test_voice_token_for_existing_chat(client, alice, auth_headers, chat_factory):
    chat = await chat_factory(alice)

    response = await client.post("/voice/token", json={"chat_id": str(chat.id)}, headers=auth_headers(alice))

    assert response.json()["chat_id"] == str(chat.id)


async def test_voice_token_for_someone_elses_chat(client, alice, user_factory, auth_headers, chat_factory):
    chat = await chat_factory(await user_factory())

    response = await client.post("/voice/token", json={"chat_id": str(chat.id)}, headers=auth_headers(alice))

    assert response.status_code == 404


async def test_voice_token_requires_auth(client):
    assert (await client.post("/voice/token", json={})).status_code == 401


@pytest.mark.parametrize("missing", ["livekit_api_key", "livekit_api_secret"])
async def test_voice_disabled_without_livekit_credentials(client, alice, auth_headers, settings, missing):
    setattr(settings, missing, None)

    response = await client.post("/voice/token", json={}, headers=auth_headers(alice))

    assert response.status_code == 503
    assert (await client.get("/chats", headers=auth_headers(alice))).json() == []  # nothing created


# ------------------------------------------------------------- app & static
async def test_health(client):
    response = await client.get("/health")
    assert (response.status_code, response.json()) == (200, {"status": "ok"})


async def test_voice_ui_is_served(client):
    response = await client.get("/voice-ui/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


# ----------------------------------------------------------------- lifespan
@pytest.fixture
def fake_runtime(monkeypatch, settings):
    runtime = MagicMock(graph="graph", messages_store="store", user_memory="memory")
    runtime.aclose = AsyncMock()
    engine = MagicMock()
    engine.dispose = AsyncMock()
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "build_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(main, "build_engine", MagicMock(return_value=engine))
    monkeypatch.setattr(main, "build_sessionmaker", MagicMock(return_value="sessionmaker"))
    return runtime, engine


async def test_lifespan_sets_state_and_cleans_up(fake_runtime, settings):
    runtime, engine = fake_runtime
    app = FastAPI()

    async with main.lifespan(app):
        assert app.state.graph == "graph"
        assert app.state.messages_store == "store"
        assert app.state.user_memory == "memory"
        assert app.state.sessionmaker == "sessionmaker"
        main.build_engine.assert_called_once_with(settings.db_url)

    runtime.aclose.assert_awaited_once()
    engine.dispose.assert_awaited_once()


async def test_lifespan_disposes_db_even_if_runtime_close_fails(fake_runtime):
    runtime, engine = fake_runtime
    runtime.aclose.side_effect = RuntimeError("mem0 close failed")

    with pytest.raises(RuntimeError):
        async with main.lifespan(FastAPI()):
            pass

    engine.dispose.assert_awaited_once()


def test_routes_are_registered():
    paths = set(main.app.openapi()["paths"])
    assert {
        "/health", "/auth/token", "/auth/register", "/auth/users/me",
        "/chats", "/chats/{chat_id}", "/chats/{chat_id}/messages", "/chats/messages",
        "/voice/token",
    } <= paths  # fmt: skip


def test_build_engine_pool_settings():
    from langgraph_chatbot.db import build_engine

    engine = build_engine("postgresql+asyncpg://u:p@localhost/db")

    assert engine.pool.size() == 10
    assert engine.pool._max_overflow == 5
    assert engine.pool._pre_ping is True
    assert engine.pool._recycle == 1800


def test_get_settings_is_cached(monkeypatch, settings):
    from langgraph_chatbot import config

    monkeypatch.setattr(config, "Settings", lambda: settings)
    config.get_settings.cache_clear()
    try:
        assert config.get_settings() is config.get_settings() is settings
    finally:
        config.get_settings.cache_clear()


@pytest.mark.xfail(
    strict=True,
    reason="BUG (latent, not used by any route yet): get_current_active_user raises "
    "'Inactive user' for ACTIVE users - the condition is inverted",
)
async def test_get_current_active_user_dependency(alice):
    from langgraph_chatbot.deps import get_current_active_user

    assert await get_current_active_user(alice) is alice

