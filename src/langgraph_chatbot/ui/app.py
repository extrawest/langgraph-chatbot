"""Gradio UI: login/register + ChatGPT-like layout (chat list on the left).

Run:  poetry run python -m langgraph_chatbot.ui.app
Env:  API_URL (default http://127.0.0.1:8000)

Layout and event wiring only; handler logic lives in handlers.py, pure helpers in helpers.py.
"""

import inspect
import logging
import os

import gradio as gr

from langgraph_chatbot.ui.api_client import ApiClient
from langgraph_chatbot.ui.chat_list import render_chat_list
from langgraph_chatbot.ui.handlers import UiHandlers
from langgraph_chatbot.ui.helpers import voice_listener_js

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Gradio 5 needs type="messages"; Gradio 6 dropped the argument (messages only).
_CHATBOT_KWARGS = (
    {"type": "messages"}
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters
    else {}
)
# chat list: flat rows like a sidebar menu, only the list scrolls (not the page)
CSS = """
#chat-list { gap: 0 !important; padding: 4px !important; border-radius: 12px;
             border: 1px solid var(--border-color-primary) !important;
             background: var(--background-fill-secondary) !important;
             max-height: calc(100vh - 260px); min-height: 160px; overflow-y: auto; }
.chat-row { flex-wrap: nowrap !important; align-items: center; gap: 0 !important;
            padding: 0 !important; border-radius: 8px; }
.chat-row:hover { background: var(--table-row-focus); }
.chat-row button { background: transparent !important; border: none !important;
                   box-shadow: none !important; }
/* display:block is needed for text-overflow: ellipsis to work on a button */
.chat-open { display: block; text-align: left; font-weight: 400;
             overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.chat-open.primary { background: var(--table-row-focus) !important; font-weight: 500; }
.row-action { opacity: 0; }
.chat-row:hover .row-action { opacity: 1; }
"""
STORAGE_SECRET = os.environ.get("UI_STORAGE_SECRET", "dev-only-change-me")


def build_ui(api_url: str = API_URL) -> gr.Blocks:
    api = ApiClient(api_url)
    h = UiHandlers(api, api_url)

    with gr.Blocks(title="LangGraph Chatbot") as demo:
        # per-browser-session state
        token = gr.State(None)
        saved_token = gr.BrowserState(
            None, storage_key="chatbot_token", secret=STORAGE_SECRET
        )
        histories = gr.State({})  # chat_id -> list of messages (placeholder, in memory)
        chats_state = gr.State([])
        current_chat = gr.State(None)
        edit_target = gr.State(None)  # {"mode": "rename"|"delete", "id": chat_id}
        voice_timer = gr.Timer(2.0, active=False)

        # ---------------------------------------------------------------- auth
        with gr.Column(visible=True) as auth_view:
            gr.Markdown("# 💬 LangGraph Chatbot")
            with gr.Tabs():
                with gr.Tab("Log in"):
                    login_email = gr.Textbox(label="Email")
                    login_password = gr.Textbox(label="Password", type="password")
                    login_btn = gr.Button("Log in", variant="primary")
                with gr.Tab("Register"):
                    reg_email = gr.Textbox(label="Email")
                    reg_password = gr.Textbox(
                        label="Password", type="password", info="At least 8 characters"
                    )
                    reg_btn = gr.Button("Create account")
            auth_msg = gr.Markdown()

        # ---------------------------------------------------------------- chats
        # filled in below (after the chatbot exists); the chat list rows read it
        # only when they are rendered, i.e. after the layout is built
        SIDEBAR_OUTPUTS: list = []
        EDIT_OUTPUTS: list = []
        with gr.Row(visible=False) as chat_view:
            with gr.Column(scale=1, min_width=280):
                user_md = gr.Markdown()
                new_chat_btn = gr.Button("＋ New chat", variant="primary")
                with gr.Column(elem_id="chat-list"):
                    render_chat_list(
                        h=h,
                        token=token,
                        chats_state=chats_state,
                        current_chat=current_chat,
                        histories=histories,
                        sidebar_outputs=SIDEBAR_OUTPUTS,
                        edit_outputs=EDIT_OUTPUTS,
                    )
                # rename / delete panel: filled and shown by the buttons of a chat row
                with gr.Column(visible=False, elem_id="chat-edit") as edit_panel:
                    edit_label = gr.Markdown()
                    edit_box = gr.Textbox(show_label=False, container=False)
                    with gr.Row():
                        edit_ok = gr.Button(
                            "✓ Confirm", variant="primary", min_width=60
                        )
                        edit_cancel = gr.Button("Cancel", min_width=60)
                logout_btn = gr.Button("Log out", size="sm")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=600,
                    placeholder="Create a chat or pick one on the left",
                    **_CHATBOT_KWARGS,
                )
                with gr.Column(visible=False) as voice_panel:
                    voice_frame = gr.HTML()
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type a message and press Enter",
                        show_label=False,
                        container=False,
                        submit_btn=True,
                        scale=9,
                    )
                    voice_btn = gr.Button(
                        "🎙", variant="secondary", scale=1, min_width=30
                    )
                # clicked from JS when the voice frame says "voice-closed"; it must live
                # outside voice_panel (a hidden Column is not in the DOM)
                voice_closed = gr.Button(visible="hidden", elem_id="voice-closed")

        SIDEBAR_OUTPUTS[:] = [chats_state, current_chat, chatbot]
        EDIT_OUTPUTS[:] = [edit_target, edit_panel, edit_label, edit_box]
        SESSION_OUTPUTS = [
            token, histories, auth_view, chat_view, auth_msg, user_md,
            *SIDEBAR_OUTPUTS,
        ]  # fmt: skip

        login_args = {
            "fn": h.login,
            "inputs": [login_email, login_password],
            "outputs": SESSION_OUTPUTS,
        }
        login_btn.click(**login_args).then(lambda t: t, token, saved_token)
        login_password.submit(**login_args).then(lambda t: t, token, saved_token)
        reg_btn.click(h.register, [reg_email, reg_password], [auth_msg])
        logout_btn.click(h.logout, None, SESSION_OUTPUTS).then(
            lambda: None, None, saved_token
        )

        new_chat_btn.click(h.new_chat, [chats_state, histories], SIDEBAR_OUTPUTS)
        edit_args = {
            "fn": h.confirm_edit,
            "inputs": [token, edit_target, edit_box, current_chat, histories],
            "outputs": [*SIDEBAR_OUTPUTS, histories, edit_target, edit_panel],
        }
        edit_ok.click(**edit_args)
        edit_box.submit(**edit_args)
        edit_cancel.click(h.cancel_edit, None, [edit_target, edit_panel])
        msg.submit(
            h.send,
            [token, msg, chatbot, current_chat, histories],
            [chatbot, msg, current_chat, histories],
        ).then(h.refresh_sidebar, [token, current_chat], [chats_state])
        voice_btn.click(
            h.voice_open,
            [token, current_chat],
            [
                voice_frame,
                chats_state,
                current_chat,
                voice_panel,
                voice_timer,
            ],
        )
        voice_closed.click(
            h.voice_close, None, [voice_frame, voice_panel, voice_timer]
        ).then(h.select, [token, current_chat, chats_state, histories], SIDEBAR_OUTPUTS)
        voice_timer.tick(h.poll_chat, [token, current_chat, chatbot], [chatbot])
        demo.load(h.restore, saved_token, SESSION_OUTPUTS)
        demo.load(None, None, None, js=voice_listener_js(api_url))

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("langgraph_chatbot").setLevel(logging.INFO)
    build_ui().launch(server_name="127.0.0.1", server_port=7860, css=CSS)
