"""ApiClient against a mocked transport (wire format, error mapping, SSE parsing)."""

import json

import httpx
import pytest

from langgraph_chatbot.ui.api_client import ApiClient, ApiError, _extract_detail

BASE = "http://api.test"


def make_client(handler) -> tuple[ApiClient, list[httpx.Request]]:
    """ApiClient whose HTTP traffic goes to `handler`; returns the recorded requests."""
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    api = ApiClient(BASE + "/")
    api._client = httpx.AsyncClient(base_url=BASE, transport=httpx.MockTransport(recording))
    return api, seen


def json_response(status=200, body=None):
    return lambda request: httpx.Response(status, json=body)


# ---------------------------------------------------------- _extract_detail
@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (httpx.Response(404, json={"detail": "Chat not found"}), "Chat not found"),
        (
            httpx.Response(422, json={"detail": [{"msg": "field required"}, {"msg": "too short"}, "raw"]}),
            "field required; too short; raw",
        ),
        (httpx.Response(502, text="Bad gateway from proxy"), "Bad gateway from proxy"),
        (httpx.Response(500, text=""), "Internal Server Error"),
        (httpx.Response(400, json={"error": "x"}), "Bad Request"),
        (httpx.Response(400, json=["not", "a", "dict"]), "Bad Request"),
        (httpx.Response(400, json={"detail": {"nested": 1}}), "Bad Request"),
    ],
)
def test_extract_detail(response, detail):
    assert _extract_detail(response) == detail


# ------------------------------------------------------------------ basics
def test_base_url_trailing_slash_is_stripped():
    assert ApiClient("http://x/")._base_url == "http://x"


def test_http_client_is_created_lazily_once():
    api = ApiClient(BASE, timeout=3.0)
    assert api._client is None

    client = api._get_client()

    assert api._get_client() is client
    assert str(client.base_url) == BASE
    assert client.timeout.connect == 3.0
    assert client.timeout.read == 60.0


async def test_token_is_sent_as_bearer():
    api, seen = make_client(json_response(body=[]))

    await api.list_chats("tok")

    assert seen[0].headers["Authorization"] == "Bearer tok"
    assert seen[0].url.path == "/chats"


async def test_no_auth_header_without_token():
    api, seen = make_client(json_response(body={"access_token": "t"}))
    await api.login("a@b.c", "pw")
    assert "Authorization" not in seen[0].headers


async def test_error_response_raises_api_error():
    api, _ = make_client(json_response(404, {"detail": "Chat not found"}))

    with pytest.raises(ApiError) as err:
        await api.get_messages("tok", "c1")

    assert err.value.status_code == 404
    assert err.value.detail == "Chat not found"
    assert str(err.value) == "Chat not found"


async def test_unreachable_backend_is_status_zero():
    def down(request):
        raise httpx.ConnectError("refused")

    api, _ = make_client(down)

    with pytest.raises(ApiError) as err:
        await api.me("tok")

    assert err.value.status_code == 0
    assert BASE in err.value.detail


async def test_empty_body_returns_none():
    api, seen = make_client(lambda r: httpx.Response(204))

    assert await api.delete_chat("tok", "c1") is None
    assert (seen[0].method, seen[0].url.path) == ("DELETE", "/chats/c1")


# --------------------------------------------------------------- endpoints
async def test_login_uses_oauth2_form_fields():
    api, seen = make_client(json_response(body={"access_token": "jwt", "token_type": "bearer"}))

    assert await api.login("me@x.io", "pw") == "jwt"

    request = seen[0]
    assert (request.method, request.url.path) == ("POST", "/auth/token")
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert request.content == b"username=me%40x.io&password=pw"


async def test_register_returns_token():
    api, seen = make_client(json_response(201, {"access_token": "new-jwt"}))

    assert await api.register("me@x.io", "pw") == "new-jwt"
    assert json.loads(seen[0].content) == {"email": "me@x.io", "password": "pw"}


@pytest.mark.parametrize(
    ("call", "method", "path", "body"),
    [
        (lambda api: api.me("t"), "GET", "/auth/users/me", None),
        (lambda api: api.create_chat("t", "Title"), "POST", "/chats", {"title": "Title"}),
        (lambda api: api.rename_chat("t", "c1", "New"), "PATCH", "/chats/c1", {"title": "New"}),
        (lambda api: api.get_messages("t", "c1"), "GET", "/chats/c1/messages", None),
    ],
)
async def test_endpoint_wire_format(call, method, path, body):
    api, seen = make_client(json_response(body={}))

    await call(api)

    assert (seen[0].method, seen[0].url.path) == (method, path)
    if body is not None:
        assert json.loads(seen[0].content) == body


async def test_get_messages_from_filter_param():
    api, seen = make_client(json_response(body=[]))

    await api.get_messages("t", "c1")
    await api.get_messages("t", "c1", from_filter=5)

    assert seen[0].url.query == b""
    assert seen[1].url.params["from_filter"] == "5"


async def test_start_voice_session_hits_voice_ui():
    api, seen = make_client(lambda r: httpx.Response(200, text="<html>"))

    response = await api.start_voice_session(chat_id="c1", token="t")

    assert response.status_code == 200
    assert seen[0].url.path == "/voice-ui"


# --------------------------------------------------------------- streaming
SSE_BODY = (
    'event: chat\ndata: {"chat_id": "c1", "title": "Hi"}\n\n'
    'event: token\ndata: {"text": "Hel"}\n\n'
    ": keep-alive comment\n\n"
    'event: token\ndata: {"text": "lo"}\n\n'
    "event: done\ndata: {}\n\n"
)


async def test_stream_message_parses_sse_events():
    api, seen = make_client(
        lambda r: httpx.Response(200, text=SSE_BODY, headers={"content-type": "text/event-stream"})
    )

    events = [e async for e in api.stream_message("hello", None, token="tok")]

    assert events == [
        ("chat", {"chat_id": "c1", "title": "Hi"}),
        ("token", {"text": "Hel"}),
        ("token", {"text": "lo"}),
        ("done", {}),
    ]
    assert json.loads(seen[0].content) == {"message": "hello", "chat_id": None}
    assert seen[0].headers["Authorization"] == "Bearer tok"


async def test_stream_message_data_without_event_name():
    api, _ = make_client(lambda r: httpx.Response(200, text='data: {"x": 1}\n\n'))
    assert [e async for e in api.stream_message("m", "c1")] == [(None, {"x": 1})]


async def test_stream_message_error_status():
    api, _ = make_client(json_response(404, {"detail": "Chat not found"}))

    with pytest.raises(ApiError) as err:
        [e async for e in api.stream_message("m", "missing", token="t")]

    assert (err.value.status_code, err.value.detail) == (404, "Chat not found")


async def test_stream_message_unreachable():
    def down(request):
        raise httpx.ConnectError("refused")

    api, _ = make_client(down)

    with pytest.raises(ApiError) as err:
        [e async for e in api.stream_message("m", None)]

    assert err.value.status_code == 0
