"""Gradio UI: login/register + ChatGPT-like layout (chat list on the left).

Run:  poetry run python -m langgraph_chatbot.ui.app
Env:  API_URL (default http://127.0.0.1:8000)

MVP scope: auth and chat management go to the real backend; the assistant
reply is a placeholder echo - replace `on_send` internals with the streaming
LangGraph endpoint later.
"""

import asyncio
import inspect
import os

import gradio as gr

from langgraph_chatbot.ui.api_client import ApiClient, ApiError

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
DEFAULT_CHAT_TITLE = "New chat"

# Gradio 5 needs type="messages"; Gradio 6 dropped the argument (messages only).
_CHATBOT_KWARGS = (
    {"type": "messages"}
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters
    else {}
)
# "leave this output untouched"
_NOOP = gr.skip if hasattr(gr, "skip") else gr.update


def _noops(count: int) -> tuple:
    return tuple(_NOOP() for _ in range(count))


def _sidebar(chats: list[dict], selected: str | None, histories: dict) -> tuple:
    """Single place that renders sidebar + chat window for a given state.

    Order matches SIDEBAR_OUTPUTS: chats_state, chat_list, current_chat,
    rename_box, chatbot.
    """
    ids = {chat["id"] for chat in chats}
    if selected not in ids:
        selected = chats[0]["id"] if chats else None
    title = next((c["title"] for c in chats if c["id"] == selected), "")
    return (
        chats,
        gr.Radio(choices=[(c["title"], c["id"]) for c in chats], value=selected),
        selected,
        title,
        histories.get(selected, []) if selected else [],
    )


def build_ui(api_url: str = API_URL) -> gr.Blocks:
    api = ApiClient(api_url)

    with gr.Blocks(title="LangGraph Chatbot") as demo:
        # per-browser-session state
        token = gr.State(None)
        histories = gr.State({})  # chat_id -> list of messages (placeholder, in memory)
        chats_state = gr.State([])
        current_chat = gr.State(None)

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
                msg = gr.Textbox(
                    placeholder="Type a message and press Enter",
                    show_label=False,
                    container=False,
                )

        SIDEBAR_OUTPUTS = [chats_state, chat_list, current_chat, rename_box, chatbot]
        SESSION_OUTPUTS = [
            token, histories, auth_view, chat_view, auth_msg, user_md,
            *SIDEBAR_OUTPUTS,
        ]  # fmt: skip

        # ------------------------------------------------------------ handlers
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

        async def on_new_chat(access_token: str, hist: dict):
            try:
                chat = await api.create_chat(access_token, DEFAULT_CHAT_TITLE)
                chats = await api.list_chats(access_token)
            except ApiError as exc:
                gr.Warning(exc.detail)
                return _noops(5)
            return _sidebar(chats, chat["id"], hist)

        def on_select(chat_id: str | None, chats: list[dict], hist: dict):
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

        async def on_send(message: str, history: list, chat_id: str | None, hist: dict):
            message = (message or "").strip()
            if not message:
                return
            if not chat_id:
                gr.Warning("Create or pick a chat first")
                return

            history = [*history, {"role": "user", "content": message}]
            hist = {**hist, chat_id: history}
            yield history, "", hist

            # --- placeholder instead of the LangGraph call -------------------
            # Here goes: async for token in <SSE /chats/{id}/messages>: ...
            reply = f"Echo: {message}"
            history = [*history, {"role": "assistant", "content": ""}]
            for end in range(3, len(reply) + 3, 3):
                history[-1] = {"role": "assistant", "content": reply[:end]}
                yield history, "", {**hist, chat_id: history}
                await asyncio.sleep(0.03)

        # -------------------------------------------------------------- wiring
        login_args = {
            "fn": do_login,
            "inputs": [login_email, login_password],
            "outputs": SESSION_OUTPUTS,
        }
        login_btn.click(**login_args)
        login_password.submit(**login_args)
        reg_btn.click(do_register, [reg_email, reg_password], [auth_msg])
        logout_btn.click(do_logout, None, SESSION_OUTPUTS)

        new_chat_btn.click(on_new_chat, [token, histories], SIDEBAR_OUTPUTS)
        chat_list.input(on_select, [chat_list, chats_state, histories], SIDEBAR_OUTPUTS)
        rename_btn.click(
            on_rename, [token, current_chat, rename_box, histories], SIDEBAR_OUTPUTS
        )
        delete_btn.click(
            on_delete, [token, current_chat, histories], [*SIDEBAR_OUTPUTS, histories]
        )
        msg.submit(
            on_send, [msg, chatbot, current_chat, histories], [chatbot, msg, histories]
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
