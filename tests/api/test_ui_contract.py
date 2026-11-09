"""Contract tests: the UI's ApiClient talking to the real FastAPI app (in-process).

These catch drift between what the UI sends and what the backend accepts, which the
mocked tests on either side cannot.
"""

import httpx
import pytest
from langchain_core.messages import AIMessageChunk

from langgraph_chatbot.ui.api_client import ApiClient, ApiError


@pytest.fixture
async def api(app):
    api = ApiClient("http://test")
    api._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    yield api
    await api._client.aclose()


@pytest.fixture
async def token(api):
    return await api.register("ui@example.com", "pw-123456")


async def test_register_login_me(api, token):
    assert (await api.me(token))["email"] == "ui@example.com"

    login_token = await api.login("ui@example.com", "pw-123456")

    assert (await api.me(login_token))["email"] == "ui@example.com"


async def test_login_error_detail(api, token):
    with pytest.raises(ApiError) as err:
        await api.login("ui@example.com", "wrong")
    assert (err.value.status_code, err.value.detail) == (401, "Incorrect username or password")


async def test_validation_error_detail_is_readable(api):
    with pytest.raises(ApiError) as err:
        await api.register("not-an-email", "pw")
    assert err.value.status_code == 422
    assert "email" in err.value.detail.lower()


async def test_chat_lifecycle(api, token):
    chat = await api.create_chat(token, "Draft")
    renamed = await api.rename_chat(token, chat["id"], "Final")

    assert renamed == {"id": chat["id"], "title": "Final"}
    assert await api.list_chats(token) == [renamed]
    assert await api.get_messages(token, chat["id"]) == []


async def test_stream_message(api, token, fake_graph):
    fake_graph.script = [(AIMessageChunk("Hi"), "agent"), (AIMessageChunk("!"), "agent")]

    events = [e async for e in api.stream_message("hello", None, token=token)]

    names = [name for name, _ in events]
    assert names == ["chat", "token", "token", "done"]
    assert events[0][1]["title"] == "hello"
    assert [d["text"] for n, d in events if n == "token"] == ["Hi", "!"]


async def test_stream_message_missing_chat(api, token):
    with pytest.raises(ApiError) as err:
        [e async for e in api.stream_message("hi", "00000000-0000-0000-0000-000000000000", token=token)]
    assert (err.value.status_code, err.value.detail) == (404, "Chat not found")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: the UI's delete button calls DELETE /chats/{id}, but the backend has no "
    "such route (ChatService.delete_chat exists, it is never exposed; Qdrant messages "
    "would also need MessagesStore.delete_chat)",
)
async def test_delete_chat(api, token):
    chat = await api.create_chat(token, "Doomed")

    await api.delete_chat(token, chat["id"])

    assert await api.list_chats(token) == []
