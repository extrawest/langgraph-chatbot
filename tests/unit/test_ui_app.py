"""Smoke test: the Gradio layout builds and its event wiring is consistent."""

import gradio as gr

from langgraph_chatbot.ui.app import _CHATBOT_KWARGS, build_ui


def test_build_ui():
    demo = build_ui("http://api.internal:8000", "https://chat.example.com")

    assert isinstance(demo, gr.Blocks)
    assert demo.title == "LangGraph Chatbot"
    # every handler wired in app.py is registered as a dependency
    fn_names = {getattr(dep.fn, "__name__", None) for dep in demo.fns.values()}
    assert {
        "login", "register", "logout", "new_chat", "confirm_edit", "cancel_edit",
        "send", "refresh_sidebar", "voice_open", "voice_close", "select", "poll_chat", "restore",
    } <= fn_names  # fmt: skip


def test_chatbot_kwargs_match_installed_gradio():
    assert _CHATBOT_KWARGS in ({}, {"type": "messages"})
