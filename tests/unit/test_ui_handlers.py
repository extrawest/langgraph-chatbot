"""Gradio handlers with a mocked ApiClient. Output tuples must match the wiring in app.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from langgraph_chatbot.ui import handlers as handlers_module
from langgraph_chatbot.ui.api_client import ApiClient, ApiError
from langgraph_chatbot.ui.handlers import UiHandlers
from langgraph_chatbot.ui.helpers import NOOP, SESSION_SIZE, SIDEBAR_SIZE, noops

PUBLIC_URL = "https://chat.example.com"
CHATS = [{"id": "c1", "title": "First"}, {"id": "c2", "title": "<b>Second</b>"}]


@pytest.fixture
def api():
    api = AsyncMock(spec=ApiClient)
    api.me.return_value = {"email": "me@example.com"}
    api.list_chats.return_value = CHATS
    return api


@pytest.fixture
def h(api):
    return UiHandlers(api, PUBLIC_URL)


@pytest.fixture
def warnings(monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(handlers_module.gr, "Warning", shown.append)
    return shown


def assert_logged_in(result, token="tok"):
    assert len(result) == SESSION_SIZE
    access_token, histories, auth_view, chat_view, auth_msg, user_md, *sidebar = result
    assert access_token == token
    assert histories == {}
    assert auth_view.visible is False
    assert chat_view.visible is True
    assert auth_msg == ""
    assert user_md == "👤 **me@example.com**"
    assert sidebar == [CHATS, None, []]


# -------------------------------------------------------------------- auth
async def test_restore_without_token(h, api):
    assert await h.restore(None) == noops(SESSION_SIZE)
    api.me.assert_not_awaited()


async def test_restore_with_valid_token(h):
    assert_logged_in(await h.restore("tok"))


async def test_restore_with_expired_token(h, api):
    api.me.side_effect = ApiError(401, "Could not validate credentials")
    assert await h.restore("old") == noops(SESSION_SIZE)


@pytest.mark.parametrize(("email", "password"), [("", "pw"), ("   ", "pw"), (None, "pw"), ("a@b.c", "")])
async def test_login_requires_both_fields(h, api, email, password):
    result = await h.login(email, password)

    assert len(result) == SESSION_SIZE
    assert result[4] == "Enter email and password"
    api.login.assert_not_awaited()


async def test_login_success(h, api):
    api.login.return_value = "tok"

    result = await h.login("  me@example.com ", "pw")

    api.login.assert_awaited_once_with("me@example.com", "pw")
    assert_logged_in(result)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ApiError(401, "Incorrect username or password"), "⚠️ Incorrect email or password"),
        (ApiError(0, "Backend is unreachable"), "⚠️ Backend is unreachable"),
    ],
)
async def test_login_failure(h, api, error, message):
    api.login.side_effect = error

    result = await h.login("me@example.com", "pw")

    assert len(result) == SESSION_SIZE
    assert result[4] == message
    assert result[0] == NOOP()  # token untouched


async def test_register_success_logs_in(h, api):
    api.register.return_value = "tok"

    result = await h.register(" me@example.com ", "pw")

    api.register.assert_awaited_once_with("me@example.com", "pw")
    assert_logged_in(result)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: register() is wired to SESSION_OUTPUTS (9 outputs) in app.py but its "
    "error path returns a single string, so Gradio errors instead of showing the message",
)
async def test_register_failure_matches_session_outputs(h, api):
    api.register.side_effect = ApiError(409, "Email already registered")

    result = await h.register("me@example.com", "pw")

    assert isinstance(result, tuple) and len(result) == SESSION_SIZE
    assert result[4] == "⚠️ Email already registered"


def test_logout(h):
    token, histories, auth_view, chat_view, auth_msg, user_md, *sidebar = h.logout()

    assert (token, histories, auth_msg, user_md) == (None, {}, "", "")
    assert auth_view.visible is True
    assert chat_view.visible is False
    assert sidebar == [[], None, []]


# ------------------------------------------------------------------- chats
def test_new_chat_clears_selection(h):
    assert h.new_chat(CHATS, {"c1": ["x"]}) == (CHATS, None, [])


async def test_select_loads_messages_from_backend(h, api):
    api.get_messages.return_value = [{"role": "user", "content": "hi"}]

    result = await h.select("tok", "c1", CHATS, {"c1": ["stale"]})

    api.get_messages.assert_awaited_once_with("tok", "c1")
    assert result == (CHATS, "c1", [{"role": "user", "content": "hi"}])


async def test_select_none_is_a_new_chat(h, api):
    assert await h.select("tok", None, CHATS, {}) == (CHATS, None, [])
    api.get_messages.assert_not_awaited()


async def test_select_failure_warns_and_keeps_ui(h, api, warnings):
    api.get_messages.side_effect = ApiError(404, "Chat not found")

    assert await h.select("tok", "c1", CHATS, {}) == noops(SIDEBAR_SIZE)
    assert warnings == ["Chat not found"]


def test_begin_rename_prefills_title(h):
    target, panel, label, box = h.begin_edit("rename", "c1", CHATS)

    assert target == {"mode": "rename", "id": "c1"}
    assert panel.visible is True
    assert label.value == "**Rename chat**"
    assert (box.value, box.visible) == ("First", True)


def test_begin_delete_escapes_title(h):
    target, panel, label, box = h.begin_edit("delete", "c2", CHATS)

    assert target == {"mode": "delete", "id": "c2"}
    assert label.value == "**Delete “&lt;b&gt;Second&lt;/b&gt;”?**"
    assert box.visible is False


def test_begin_edit_unknown_chat(h):
    _, _, _, box = h.begin_edit("rename", "gone", CHATS)
    assert box.value == ""


def test_cancel_edit(h):
    target, panel = h.cancel_edit()
    assert target is None
    assert panel.visible is False


async def test_confirm_edit_without_target(h, api):
    assert await h.confirm_edit("tok", None, "x", None, {}) == noops(SIDEBAR_SIZE + 3)
    api.rename_chat.assert_not_awaited()


async def test_confirm_rename(h, api):
    hist = {"c1": ["m"]}

    result = await h.confirm_edit("tok", {"mode": "rename", "id": "c1"}, "  New name ", "c1", hist)

    api.rename_chat.assert_awaited_once_with("tok", "c1", "New name")
    chats, current, chatbot, histories, target, panel = result
    assert (chats, current, chatbot, histories, target) == (CHATS, "c1", ["m"], hist, None)
    assert panel.visible is False


async def test_confirm_rename_rejects_blank_title(h, api, warnings):
    result = await h.confirm_edit("tok", {"mode": "rename", "id": "c1"}, "   ", None, {})

    assert result == noops(SIDEBAR_SIZE + 3)  # panel stays open
    assert warnings == ["Enter a non-empty title"]
    api.rename_chat.assert_not_awaited()


async def test_confirm_rename_api_error(h, api, warnings):
    api.rename_chat.side_effect = ApiError(404, "Chat not found")

    result = await h.confirm_edit("tok", {"mode": "rename", "id": "c1"}, "x", None, {})

    assert result == noops(SIDEBAR_SIZE + 3)
    assert warnings == ["Chat not found"]


async def test_confirm_delete_of_open_chat_clears_selection(h, api):
    api.list_chats.return_value = [CHATS[1]]
    hist = {"c1": ["a"], "c2": ["b"]}

    result = await h.confirm_edit("tok", {"mode": "delete", "id": "c1"}, "", "c1", hist)

    api.delete_chat.assert_awaited_once_with("tok", "c1")
    chats, current, chatbot, histories, target, _ = result
    assert (chats, current, chatbot) == ([CHATS[1]], None, [])
    assert histories == {"c2": ["b"]}
    assert target is None


async def test_confirm_delete_of_other_chat_keeps_open_one(h, api):
    api.list_chats.return_value = [CHATS[1]]

    result = await h.confirm_edit("tok", {"mode": "delete", "id": "c1"}, "", "c2", {"c2": ["b"]})

    assert result[:3] == ([CHATS[1]], "c2", ["b"])


async def test_confirm_delete_api_error(h, api, warnings):
    api.delete_chat.side_effect = ApiError(405, "Method Not Allowed")

    result = await h.confirm_edit("tok", {"mode": "delete", "id": "c1"}, "", "c1", {})

    assert result == noops(SIDEBAR_SIZE + 3)
    assert warnings == ["Method Not Allowed"]


async def test_refresh_sidebar(h, api):
    assert await h.refresh_sidebar("tok", "c1") == CHATS


# ---------------------------------------------------------------- messages
def stream_of(*events, error: ApiError | None = None):
    async def _stream(message, chat_id, *, token=None):
        for event in events:
            yield event
        if error:
            raise error

    return MagicMock(side_effect=_stream)


async def run_send(h, *args):
    return [tuple(map(_copy, out)) async for out in h.send(*args)]


def _copy(value):
    # handlers mutate `history` in place between yields; snapshot each output
    import copy

    return copy.deepcopy(value)


async def test_send_ignores_blank_message(h, api):
    outputs = await run_send(h, "tok", "   ", ["old"], "c1", {})

    assert outputs == [(["old"], "", "c1", {})]
    api.stream_message.assert_not_called()


async def test_send_streams_answer_into_new_chat(h, api):
    api.stream_message = stream_of(
        ("chat", {"chat_id": "new-id", "title": "Hi"}),
        ("token", {"text": "Hel"}),
        ("token", {"text": "lo"}),
        ("done", {}),
        ("token", {"text": " ignored after done"}),
    )

    outputs = await run_send(h, "tok", "  hi  ", [], None, {})

    api.stream_message.assert_called_once_with("hi", None, token="tok")
    # first yield shows the question and an empty answer bubble, input cleared
    assert outputs[0] == ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}], "", None, {})
    final_history, msg, chat_id, hist = outputs[-1]
    assert final_history[-1] == {"role": "assistant", "content": "Hello"}
    assert (msg, chat_id) == ("", "new-id")
    assert hist == {"new-id": final_history}


async def test_send_shows_tool_progress(h, api):
    api.stream_message = stream_of(
        ("chat", {"chat_id": "c1"}),
        ("tool", {"name": "tavily_search", "status": "started"}),
        ("tool", {"name": "tavily_search", "status": "done"}),
        ("token", {"text": "Answer"}),
        ("done", {}),
    )

    outputs = await run_send(h, "tok", "q", [], "c1", {})

    started = outputs[1][0]  # [0] = question echo; the "chat" event yields nothing
    assert started[1]["metadata"] == {"title": "tavily_search", "status": "pending"}
    final = outputs[-1][0]
    assert [m.get("metadata", {}).get("status") for m in final] == [None, "done", None]
    assert final[-1]["content"] == "Answer"


async def test_send_appends_stream_error(h, api):
    api.stream_message = stream_of(
        ("chat", {"chat_id": "c1"}),
        ("token", {"text": "partial"}),
        ("error", {"detail": "Something went wrong"}),
    )

    outputs = await run_send(h, "tok", "q", [], "c1", {})

    assert outputs[-1][0][-1]["content"] == "partial\n\n⚠️ Something went wrong"


async def test_send_api_error_replaces_answer(h, api):
    api.stream_message = stream_of(error=ApiError(404, "Chat not found"))

    outputs = await run_send(h, "tok", "q", [], "missing", {})

    history, _, chat_id, hist = outputs[-1]
    assert history[-1]["content"] == "⚠️ Chat not found"
    assert (chat_id, hist) == ("missing", {})


def test_tool_done_marks_only_first_pending_of_that_name():
    history = [
        {"role": "assistant", "content": "", "metadata": {"title": "a", "status": "done"}},
        {"role": "assistant", "content": "", "metadata": {"title": "a", "status": "pending"}},
        {"role": "assistant", "content": "", "metadata": {"title": "a", "status": "pending"}},
        {"role": "assistant", "content": ""},
    ]

    UiHandlers._apply_tool_event(history, {"name": "a", "status": "done"})

    assert [m.get("metadata", {}).get("status") for m in history] == ["done", "done", "pending", None]


def test_tool_done_without_matching_start_is_ignored():
    history = [{"role": "assistant", "content": ""}]
    UiHandlers._apply_tool_event(history, {"name": "x", "status": "done"})
    assert history == [{"role": "assistant", "content": ""}]


# ------------------------------------------------------------------- voice
async def test_voice_open_requires_login(h, warnings):
    assert await h.voice_open(None, "c1") == noops(5)
    assert warnings == ["Log in first"]


async def test_voice_open_existing_chat(h, api):
    frame, chats, chat_id, panel, timer = await h.voice_open("tok", "c1")

    api.create_chat.assert_not_awaited()
    assert chats == NOOP()
    assert chat_id == "c1"
    assert f'src="{PUBLIC_URL}/voice-ui/#token=tok&amp;chat_id=c1"' in frame
    assert 'allow="microphone; autoplay"' in frame
    assert panel.visible is True
    assert timer.active is True


async def test_voice_open_creates_chat_when_none_selected(h, api):
    api.create_chat.return_value = {"id": "voice-chat"}

    _, chats, chat_id, _, _ = await h.voice_open("tok", None)

    api.create_chat.assert_awaited_once_with("tok", "Voice chat")
    assert (chats, chat_id) == (CHATS, "voice-chat")


async def test_voice_open_create_failure(h, api, warnings):
    api.create_chat.side_effect = ApiError(0, "Backend is unreachable")

    assert await h.voice_open("tok", None) == noops(5)
    assert warnings == ["Backend is unreachable"]


async def test_voice_open_escapes_token_in_html(h):
    frame, *_ = await h.voice_open('a"b<c', "c1")
    assert 'a"b<c' not in frame


def test_voice_close(h):
    frame, panel, timer = h.voice_close()
    assert frame == ""
    assert panel.visible is False
    assert timer.active is False


@pytest.mark.parametrize(("token", "chat_id"), [(None, "c1"), ("tok", None)])
async def test_poll_chat_needs_token_and_chat(h, api, token, chat_id):
    assert await h.poll_chat(token, chat_id, []) == NOOP()
    api.get_messages.assert_not_awaited()


async def test_poll_chat_updates_only_on_change(h, api):
    api.get_messages.return_value = [{"role": "user", "content": "hi"}]

    assert await h.poll_chat("tok", "c1", [{"role": "user", "content": "hi"}]) == NOOP()
    assert await h.poll_chat("tok", "c1", []) == [{"role": "user", "content": "hi"}]


async def test_poll_chat_ignores_errors(h, api):
    api.get_messages.side_effect = ApiError(0, "down")
    assert await h.poll_chat("tok", "c1", []) == NOOP()
