"""Small pure helpers shared by the UI layout and its handlers."""

import json
from urllib.parse import urlsplit

import gradio as gr

# "leave this output untouched"
NOOP = gr.skip if hasattr(gr, "skip") else gr.update

# how many outputs the sidebar / a whole session update touches (see app.py)
SIDEBAR_SIZE = 3
SESSION_SIZE = 6 + SIDEBAR_SIZE


def noops(count: int) -> tuple:
    return tuple(NOOP() for _ in range(count))


def sidebar(chats: list[dict], selected: str | None, histories: dict) -> tuple:
    """Single place that renders sidebar + chat window for a given state.

    Order matches SIDEBAR_OUTPUTS: chats_state, current_chat, chatbot.
    """
    ids = {chat["id"] for chat in chats}
    if selected not in ids:
        selected = None  # no selection = "new chat" draft (created on first message)
    return (
        chats,
        selected,
        histories.get(selected, []) if selected else [],
    )


def voice_listener_js(api_url: str) -> str:
    """Page-load JS: when the voice frame posts "voice-closed", click the hidden button."""
    parts = urlsplit(api_url)
    origin = json.dumps(f"{parts.scheme}://{parts.netloc}")
    return f"""() => {{
        if (window.__voiceListener) return;
        window.__voiceListener = true;
        window.addEventListener("message", (event) => {{
            if (event.origin !== {origin}) return;
            if (!event.data || event.data.type !== "voice-closed") return;
            const el = document.getElementById("voice-closed");
            (el && el.tagName === "BUTTON" ? el : el?.querySelector("button"))?.click();
        }});
    }}"""
