import uuid

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from tests.helpers import parse_sse


@pytest.fixture
async def alice(user_factory):
    return await user_factory(email="alice@example.com")


@pytest.fixture
async def bob(user_factory):
    return await user_factory(email="bob@example.com")


@pytest.fixture
def as_alice(alice, auth_headers):
    return auth_headers(alice)


# ------------------------------------------------------------------- auth
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/chats"),
        ("POST", "/chats"),
        ("GET", f"/chats/{uuid.uuid4()}"),
        ("PATCH", f"/chats/{uuid.uuid4()}"),
        ("GET", f"/chats/{uuid.uuid4()}/messages"),
        ("POST", "/chats/messages"),
    ],
)
async def test_chat_endpoints_require_auth(client, method, path):
    response = await client.request(method, path, json={"title": "x", "message": "x"})
    assert response.status_code == 401


# ------------------------------------------------------------------- CRUD
async def test_list_starts_empty(client, as_alice):
    response = await client.get("/chats", headers=as_alice)
    assert (response.status_code, response.json()) == (200, [])


async def test_create_and_list(client, as_alice):
    created = await client.post("/chats", json={"title": "  Trip plans  "}, headers=as_alice)

    assert created.status_code == 201
    chat = created.json()
    assert set(chat) == {"id", "title"}
    assert chat["title"] == "Trip plans"
    assert (await client.get("/chats", headers=as_alice)).json() == [chat]


async def test_create_default_title(client, as_alice):
    response = await client.post("/chats", json={}, headers=as_alice)
    assert response.json()["title"] == "New chat"


@pytest.mark.parametrize("title", ["", "x" * 256])
async def test_create_validates_title(client, as_alice, title):
    assert (await client.post("/chats", json={"title": title}, headers=as_alice)).status_code == 422


@pytest.mark.parametrize("method", ["POST", "PATCH"])
async def test_whitespace_only_title_is_rejected(client, as_alice, alice, chat_factory, method):
    chat = await chat_factory(alice)
    path = "/chats" if method == "POST" else f"/chats/{chat.id}"

    response = await client.request(method, path, json={"title": "   "}, headers=as_alice)

    assert response.status_code == 422


async def test_users_see_only_their_own_chats(client, alice, bob, chat_factory, auth_headers):
    mine = await chat_factory(alice, "alice's")
    theirs = await chat_factory(bob, "bob's")

    listed = (await client.get("/chats", headers=auth_headers(alice))).json()

    assert [c["id"] for c in listed] == [str(mine.id)]
    for path in (f"/chats/{theirs.id}", f"/chats/{theirs.id}/messages"):
        response = await client.get(path, headers=auth_headers(alice))
        assert response.status_code == 404, path
        assert response.json() == {"detail": "Chat not found"}
    response = await client.patch(f"/chats/{theirs.id}", json={"title": "pwned"}, headers=auth_headers(alice))
    assert response.status_code == 404


async def test_get_chat(client, as_alice, alice, chat_factory):
    chat = await chat_factory(alice, "Mine")

    response = await client.get(f"/chats/{chat.id}", headers=as_alice)

    assert response.json() == {"id": str(chat.id), "title": "Mine"}


async def test_get_chat_bad_id(client, as_alice):
    assert (await client.get("/chats/not-a-uuid", headers=as_alice)).status_code == 422
    assert (await client.get(f"/chats/{uuid.uuid4()}", headers=as_alice)).status_code == 404


async def test_rename(client, as_alice, alice, chat_factory):
    chat = await chat_factory(alice, "old")

    response = await client.patch(f"/chats/{chat.id}", json={"title": " new "}, headers=as_alice)

    assert response.status_code == 200
    assert response.json() == {"id": str(chat.id), "title": "new"}


async def test_rename_requires_title(client, as_alice, alice, chat_factory):
    chat = await chat_factory(alice)
    assert (await client.patch(f"/chats/{chat.id}", json={}, headers=as_alice)).status_code == 422


async def test_messages(client, as_alice, alice, chat_factory, messages_store):
    chat = await chat_factory(alice)
    assert (await client.get(f"/chats/{chat.id}/messages", headers=as_alice)).json() == []

    await messages_store.add(user_id=alice.id, chat_id=chat.id, turn_id="t", role="user", content="hi")

    response = await client.get(f"/chats/{chat.id}/messages", headers=as_alice)
    assert response.json() == [{"role": "user", "content": "hi"}]


async def test_whitespace_title_is_stripped_before_saving(client, as_alice):
    response = await client.post("/chats", json={"title": " \t Notes \n"}, headers=as_alice)
    assert response.json()["title"] == "Notes"


# --------------------------------------------------------- send (SSE stream)
async def test_send_message_to_new_chat_creates_it(client, as_alice, alice, fake_graph):
    fake_graph.script = [
        (AIMessageChunk("", tool_call_chunks=[{"name": "tavily_search", "args": "", "id": "1", "index": 0}]), "agent"),
        (ToolMessage("results", tool_call_id="1", name="tavily_search"), "agent_tools"),
        (AIMessageChunk("Hello"), "agent"),
        (AIMessageChunk(" there"), "agent"),
    ]

    response = await client.post(
        "/chats/messages",
        json={"message": "  What's the weather like in Paris this weekend?  "},
        headers=as_alice,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    (name, chat), *rest = events
    assert name == "chat"
    assert chat["title"] == "What's the weather like in Paris this…"
    assert rest == [
        ("tool", {"name": "tavily_search", "status": "started"}),
        ("tool", {"name": "tavily_search", "status": "done"}),
        ("token", {"text": "Hello"}),
        ("token", {"text": " there"}),
        ("done", {}),
    ]
    # the chat really exists now and the graph got its id
    listed = (await client.get("/chats", headers=as_alice)).json()
    assert listed == [{"id": chat["chat_id"], "title": chat["title"]}]
    assert str(fake_graph.calls[0]["context"].chat_id) == chat["chat_id"]
    assert fake_graph.calls[0]["context"].user_id == alice.id


async def test_send_message_to_existing_chat(client, as_alice, alice, chat_factory, fake_graph):
    chat = await chat_factory(alice, "Existing")
    fake_graph.script = [(AIMessageChunk("ok"), "agent")]

    response = await client.post(
        "/chats/messages", json={"message": "hi", "chat_id": str(chat.id)}, headers=as_alice
    )

    events = parse_sse(response.text)
    assert events[0] == ("chat", {"chat_id": str(chat.id), "title": "Existing"})
    assert len((await client.get("/chats", headers=as_alice)).json()) == 1


async def test_send_message_to_missing_chat_is_a_plain_404(client, as_alice, fake_graph):
    response = await client.post(
        "/chats/messages", json={"message": "hi", "chat_id": str(uuid.uuid4())}, headers=as_alice
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Chat not found"}
    assert fake_graph.calls == []


async def test_send_message_to_someone_elses_chat(client, bob, chat_factory, as_alice, fake_graph):
    chat = await chat_factory(bob)

    response = await client.post(
        "/chats/messages", json={"message": "hi", "chat_id": str(chat.id)}, headers=as_alice
    )

    assert response.status_code == 404
    assert fake_graph.calls == []


async def test_send_message_graph_failure_becomes_error_event(client, as_alice, fake_graph):
    fake_graph.script = [(AIMessageChunk("par"), "agent"), TimeoutError()]

    response = await client.post("/chats/messages", json={"message": "hi"}, headers=as_alice)

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[-2:] == [
        ("token", {"text": "par"}),
        ("error", {"detail": "The request took too long, please try again"}),
    ]


async def test_send_message_validation(client, as_alice):
    assert (await client.post("/chats/messages", json={}, headers=as_alice)).status_code == 422
    response = await client.post("/chats/messages", json={"message": "x", "chat_id": "nope"}, headers=as_alice)
    assert response.status_code == 422
