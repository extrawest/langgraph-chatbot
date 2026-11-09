import json

import gradio as gr
import pytest

from langgraph_chatbot.ui.helpers import (
    NOOP,
    SESSION_SIZE,
    SIDEBAR_SIZE,
    noops,
    sidebar,
    voice_listener_js,
)

CHATS = [{"id": "c1", "title": "One"}, {"id": "c2", "title": "Two"}]


def test_noops():
    assert noops(3) == (NOOP(), NOOP(), NOOP())
    assert noops(0) == ()


def test_sizes_match_the_wiring():
    assert SIDEBAR_SIZE == 3
    assert SESSION_SIZE == 9


def test_sidebar_selects_known_chat_with_its_history():
    assert sidebar(CHATS, "c2", {"c2": ["msg"]}) == (CHATS, "c2", ["msg"])


def test_sidebar_known_chat_without_loaded_history():
    assert sidebar(CHATS, "c1", {}) == (CHATS, "c1", [])


@pytest.mark.parametrize("selected", [None, "deleted-chat"])
def test_sidebar_unknown_selection_becomes_new_chat_draft(selected):
    assert sidebar(CHATS, selected, {"deleted-chat": ["x"]}) == (CHATS, None, [])


@pytest.mark.parametrize(
    ("api_url", "origin"),
    [
        ("https://api.example.com/some/path", "https://api.example.com"),
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
    ],
)
def test_voice_listener_checks_origin(api_url, origin):
    js = voice_listener_js(api_url)

    assert f"event.origin !== {json.dumps(origin)}" in js
    assert '"voice-closed"' in js
    assert 'getElementById("voice-closed")' in js


def test_voice_listener_origin_is_js_escaped():
    js = voice_listener_js('https://evil"host')
    assert '"https://evil\\"host"' in js


def test_noop_is_a_gradio_skip_or_update():
    assert NOOP in (getattr(gr, "skip", None), gr.update)
