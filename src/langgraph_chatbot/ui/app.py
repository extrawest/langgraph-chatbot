"""Gradio UI: login/register + ChatGPT-like layout (chat list on the left).

Run:  poetry run python -m langgraph_chatbot.ui.app
Env:  API_URL (default http://127.0.0.1:8000)

MVP scope: auth and chat management go to the real backend; the assistant
reply is a placeholder echo - replace `on_send` internals with the streaming
LangGraph endpoint later.
"""

import inspect
import json
import os
from html import escape
from urllib.parse import urlencode, urlsplit

import gradio as gr

from langgraph_chatbot.ui.api_client import ApiClient, ApiError

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Gradio 5 needs type="messages"; Gradio 6 dropped the argument (messages only).
_CHATBOT_KWARGS = (
    {"type": "messages"}
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters
    else {}
)
# "leave this output untouched"
_NOOP = gr.skip if hasattr(gr, "skip") else gr.update

STORAGE_SECRET = os.environ.get("UI_STORAGE_SECRET", "dev-only-change-me")


def _noops(count: int) -> tuple:
    return tuple(_NOOP() for _ in range(count))


def _sidebar(chats: list[dict], selected: str | None, histories: dict) -> tuple:
    """Single place that renders sidebar + chat window for a given state.

    Order matches SIDEBAR_OUTPUTS: chats_state, chat_list, current_chat,
    rename_box, chatbot.
    """
    ids = {chat["id"] for chat in chats}
    if selected not in ids:
        selected = None  # no selection = "new chat" draft (created on first message)
    title = next((c["title"] for c in chats if c["id"] == selected), "")
    return (
        chats,
        gr.Radio(choices=[(c["title"], c["id"]) for c in chats], value=selected),
        selected,
        title,
        histories.get(selected, []) if selected else [],
    )


def _voice_listener_js(api_url: str) -> str:
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


def build_ui(api_url: str = API_URL) -> gr.Blocks:
    api = ApiClient(api_url)

    with gr.Blocks(title="LangGraph Chatbot") as demo:
        # per-browser-session state
        token = gr.State(None)
        saved_token = gr.BrowserState(
            None, storage_key="chatbot_token", secret=STORAGE_SECRET
        )
        histories = gr.State({})  # chat_id -> list of messages (placeholder, in memory)
        chats_state = gr.State([])
        current_chat = gr.State(None)
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
        with gr.Row(visible=False) as chat_view:
            with gr.Column(scale=1, min_width=280):
                user_md = gr.Markdown()
                new_chat_btn = gr.Button("＋ New chat", variant="primary")
                chat_list = gr.Radio(choices=[], label="Your chats", interactive=True)
                with gr.Accordion("Manage chat", open=False):
                    rename_box = gr.Textbox(label="Title")
                    with gr.Row():
                        rename_btn = gr.Button("Rename")
                        delete_btn = gr.Button("Delete", variant="stop")
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

        SIDEBAR_OUTPUTS = [chats_state, chat_list, current_chat, rename_box, chatbot]
        SESSION_OUTPUTS = [
            token, histories, auth_view, chat_view, auth_msg, user_md,
            *SIDEBAR_OUTPUTS,
        ]  # fmt: skip

        # ------------------------------------------------------------ handlers
        async def restore(access_token):
            if not access_token:
                return _noops(len(SESSION_OUTPUTS))

            try:
                me = await api.me(access_token)
                chats = await api.list_chats(access_token)
            except ApiError:
                return _noops(len(SESSION_OUTPUTS))
            return (
                access_token,
                {},
                gr.Column(visible=False),
                gr.Row(visible=True),
                "",
                f"👤 **{me['email']}**",
                *_sidebar(chats, None, {}),
            )

        async def do_login(email: str, password: str):
            email = (email or "").strip()
            if not email or not password:
                return (*_noops(4), "Enter email and password", *_noops(6))
            try:
                access_token = await api.login(email, password)
                me = await api.me(access_token)
                chats = await api.list_chats(access_token)
            except ApiError as exc:
                detail = (
                    "Incorrect email or password"
                    if exc.status_code == 401
                    else exc.detail
                )
                return (*_noops(4), f"⚠️ {detail}", *_noops(6))

            return (
                access_token,
                {},
                gr.Column(visible=False),
                gr.Row(visible=True),
                "",
                f"👤 **{me['email']}**",
                *_sidebar(chats, None, {}),
            )

        async def do_register(email: str, password: str):
            try:
                await api.register((email or "").strip(), password or "")
            except ApiError as exc:
                return f"⚠️ {exc.detail}"
            return "✅ Account created - now log in on the first tab"

        def do_logout():
            return (
                None,
                {},
                gr.Column(visible=True),
                gr.Row(visible=False),
                "",
                "",
                *_sidebar([], None, {}),
            )

        def on_new_chat(chats: list[dict], hist: dict):
            # nothing is created here: the chat appears on the backend with the first message
            return _sidebar(chats, None, hist)

        async def on_select(
            access_token: str, chat_id: str | None, chats: list[dict], hist: dict
        ):
            if chat_id:
                try:
                    messages = await api.get_messages(access_token, chat_id)
                except ApiError as exc:
                    gr.Warning(exc.detail)
                    return _noops(5)
                hist = {**hist, chat_id: messages}  # Qdrant is the source of truth
            return _sidebar(chats, chat_id, hist)

        async def on_rename(
            access_token: str, chat_id: str | None, title: str, hist: dict
        ):
            title = (title or "").strip()
            if not chat_id or not title:
                gr.Warning("Pick a chat and enter a non-empty title")
                return _noops(5)
            try:
                await api.rename_chat(access_token, chat_id, title)
                chats = await api.list_chats(access_token)
            except ApiError as exc:
                gr.Warning(exc.detail)
                return _noops(5)
            return _sidebar(chats, chat_id, hist)

        async def on_delete(access_token: str, chat_id: str | None, hist: dict):
            if not chat_id:
                gr.Warning("Pick a chat first")
                return _noops(6)
            try:
                await api.delete_chat(access_token, chat_id)
                chats = await api.list_chats(access_token)
            except ApiError as exc:
                gr.Warning(exc.detail)
                return _noops(6)
            hist = {k: v for k, v in hist.items() if k != chat_id}
            return (*_sidebar(chats, None, hist), hist)

        async def on_send(
            access_token, message: str, history: list, chat_id: str | None, hist: dict
        ):
            print(f"Message is: {message}")
            message = (message or "").strip()
            if not message:
                print("here")
                yield history, "", chat_id, hist
                return

            history = [
                *history,
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            yield history, "", chat_id, hist

            try:
                async for event_name, data in api.stream_message(
                    message, chat_id, token=access_token
                ):
                    if event_name == "chat":
                        chat_id = data["chat_id"]
                    elif event_name == "token":
                        history[-1]["content"] += data["text"]
                        yield history, "", chat_id, hist
                    elif event_name == "tool":
                        gr.Info(f"🔎 {data['name']}: {data['status']}")
                    elif event_name == "error":
                        history[-1]["content"] += f"\n\n⚠️ {data['detail']}"
                        yield history, "", chat_id, hist
                    elif event_name == "done":
                        break
            except ApiError as exc:
                history[-1]["content"] = f"⚠️ {exc.detail}"
                yield history, "", chat_id, hist
                return

            # chat_id is known only now (from the "chat" event) - save history once
            if chat_id:
                hist = {**hist, chat_id: history}
            yield history, "", chat_id, hist

        async def refresh_sidebar(access_token, chat_id):
            chats = await api.list_chats(access_token)
            radio = gr.Radio(
                choices=[(c["title"], c["id"]) for c in chats],
                value=chat_id,
            )
            return chats, radio

        def on_voice_open(access_token: str | None, chat_id: str | None):
            if not access_token:
                gr.Warning("Log in first")
                return _noops(3)
            # fragment (#...) is never sent to a server or written to logs
            fragment = urlencode({"token": access_token, "chat_id": chat_id or ""})
            src = escape(f"{api_url}/voice-ui/#{fragment}")
            frame = (
                f'<iframe src="{src}" allow="microphone; autoplay" '
                'style="width:100%;height:56px;border:0;border-radius:12px">'
                "</iframe>"
            )

            return frame, gr.Column(visible=True), gr.Timer(active=True)

        async def poll_chat(access_token, chat_id, history):
            # a chat without an id yet (brand-new voice chat) has nothing to poll
            if not access_token or not chat_id:
                return _NOOP()
            try:
                messages = await api.get_messages(access_token, chat_id)
            except ApiError:
                return _NOOP()
            if len(messages) == len(history):
                return _NOOP()
            return messages

        def on_voice_close():
            # removing the iframe destroys the page -> LiveKit room disconnects
            return "", gr.Column(visible=False), gr.Timer(active=False)

        login_args = {
            "fn": do_login,
            "inputs": [login_email, login_password],
            "outputs": SESSION_OUTPUTS,
        }
        login_btn.click(**login_args).then(lambda t: t, token, saved_token)
        login_password.submit(**login_args).then(lambda t: t, token, saved_token)
        reg_btn.click(do_register, [reg_email, reg_password], [auth_msg])
        logout_btn.click(do_logout, None, SESSION_OUTPUTS).then(
            lambda: None, None, saved_token
        )

        new_chat_btn.click(on_new_chat, [chats_state, histories], SIDEBAR_OUTPUTS)
        chat_list.input(
            on_select, [token, chat_list, chats_state, histories], SIDEBAR_OUTPUTS
        )
        rename_btn.click(
            on_rename, [token, current_chat, rename_box, histories], SIDEBAR_OUTPUTS
        )
        delete_btn.click(
            on_delete, [token, current_chat, histories], [*SIDEBAR_OUTPUTS, histories]
        )
        msg.submit(
            on_send,
            [token, msg, chatbot, current_chat, histories],
            [chatbot, msg, current_chat, histories],
        ).then(refresh_sidebar, [token, current_chat], [chats_state, chat_list])
        voice_btn.click(
            on_voice_open,
            [token, current_chat],
            [voice_frame, voice_panel, voice_timer],
        )
        voice_closed.click(
            on_voice_close, None, [voice_frame, voice_panel, voice_timer]
        ).then(
            on_select, [token, current_chat, chats_state, histories], SIDEBAR_OUTPUTS
        )
        voice_timer.tick(poll_chat, [token, current_chat, chatbot], [chatbot])
        demo.load(restore, saved_token, SESSION_OUTPUTS)
        demo.load(None, None, None, js=_voice_listener_js(api_url))

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
