"""Event handlers of the Gradio UI.

They hold no layout knowledge except the order of outputs, which is documented
next to each method and wired in app.py.
"""

import logging
from html import escape
from urllib.parse import urlencode

import gradio as gr

from langgraph_chatbot.ui.api_client import ApiClient, ApiError
from langgraph_chatbot.ui.helpers import (
    NOOP,
    SESSION_SIZE,
    SIDEBAR_SIZE,
    noops,
    sidebar,
)

logger = logging.getLogger(__name__)


class UiHandlers:
    def __init__(self, api: ApiClient, api_url: str) -> None:
        self._api = api
        self._api_url = api_url

    # ------------------------------------------------------------------ auth
    # outputs: SESSION_OUTPUTS (token, histories, auth_view, chat_view,
    #          auth_msg, user_md, + sidebar)
    async def restore(self, access_token):
        if not access_token:
            return noops(SESSION_SIZE)
        try:
            return await self._session(access_token)
        except ApiError:
            return noops(SESSION_SIZE)

    async def login(self, email: str, password: str):
        email = (email or "").strip()
        if not email or not password:
            return (*noops(4), "Enter email and password", *noops(SIDEBAR_SIZE + 1))
        try:
            access_token = await self._api.login(email, password)
            return await self._session(access_token)
        except ApiError as exc:
            detail = (
                "Incorrect email or password" if exc.status_code == 401 else exc.detail
            )
            return (*noops(4), f"⚠️ {detail}", *noops(SIDEBAR_SIZE + 1))

    async def register(self, email: str, password: str):
        try:
            await self._api.register((email or "").strip(), password or "")
        except ApiError as exc:
            return f"⚠️ {exc.detail}"
        return "✅ Account created - now log in on the first tab"

    def logout(self):
        return (
            None,
            {},
            gr.Column(visible=True),
            gr.Row(visible=False),
            "",
            "",
            *sidebar([], None, {}),
        )

    async def _session(self, access_token: str) -> tuple:
        """Everything the UI needs right after a successful login / restore."""
        me = await self._api.me(access_token)
        chats = await self._api.list_chats(access_token)
        return (
            access_token,
            {},
            gr.Column(visible=False),
            gr.Row(visible=True),
            "",
            f"👤 **{me['email']}**",
            *sidebar(chats, None, {}),
        )

    # ----------------------------------------------------------------- chats
    # outputs: SIDEBAR_OUTPUTS (chats_state, current_chat, chatbot)
    def new_chat(self, chats: list[dict], hist: dict):
        # nothing is created here: the chat appears on the backend with the first message
        return sidebar(chats, None, hist)

    async def select(
        self, access_token: str, chat_id: str | None, chats: list[dict], hist: dict
    ):
        if chat_id:
            try:
                messages = await self._api.get_messages(access_token, chat_id)
            except ApiError as exc:
                gr.Warning(exc.detail)
                return noops(SIDEBAR_SIZE)
            hist = {**hist, chat_id: messages}  # Qdrant is the source of truth
        return sidebar(chats, chat_id, hist)

    # --- edit panel (rename / delete), one static panel under the chat list ------
    # outputs: edit_target, edit_panel, edit_label, edit_box
    def begin_edit(self, mode: str, chat_id: str, chats: list[dict]):
        title = next((c["title"] for c in chats if c["id"] == chat_id), "")
        target = {"mode": mode, "id": chat_id}
        if mode == "rename":
            return (
                target,
                gr.Column(visible=True),
                gr.Markdown("**Rename chat**"),
                gr.Textbox(value=title, visible=True),
            )
        return (
            target,
            gr.Column(visible=True),
            gr.Markdown(f"**Delete “{escape(title)}”?**"),
            gr.Textbox(visible=False),
        )

    # outputs: edit_target, edit_panel
    def cancel_edit(self):
        return None, gr.Column(visible=False)

    # outputs: SIDEBAR_OUTPUTS + histories + edit_target + edit_panel
    async def confirm_edit(
        self,
        access_token: str,
        target: dict | None,
        new_title: str,
        current_id: str | None,
        hist: dict,
    ):
        if not target:
            return noops(SIDEBAR_SIZE + 3)
        logger.info("confirm_edit %s title=%r", target, new_title)
        if target["mode"] == "rename":
            result = await self._rename(
                access_token, target["id"], new_title, current_id, hist
            )
        else:
            result = await self._delete(access_token, target["id"], current_id, hist)
        if result is None:  # a warning was already shown, keep the panel open
            return noops(SIDEBAR_SIZE + 3)
        return (*result, None, gr.Column(visible=False))

    async def _rename(
        self,
        access_token: str,
        chat_id: str,
        title: str,
        current_id: str | None,
        hist: dict,
    ):
        """Returns SIDEBAR_OUTPUTS + histories, or None after showing a warning."""
        title = (title or "").strip()
        if not title:
            gr.Warning("Enter a non-empty title")
            return None
        try:
            await self._api.rename_chat(access_token, chat_id, title)
            chats = await self._api.list_chats(access_token)
        except ApiError as exc:
            gr.Warning(exc.detail)
            return None
        return (*sidebar(chats, current_id, hist), hist)

    async def _delete(
        self, access_token: str, chat_id: str, current_id: str | None, hist: dict
    ):
        """Returns SIDEBAR_OUTPUTS + histories, or None after showing a warning."""
        try:
            await self._api.delete_chat(access_token, chat_id)
            chats = await self._api.list_chats(access_token)
        except ApiError as exc:
            gr.Warning(exc.detail)
            return None
        hist = {k: v for k, v in hist.items() if k != chat_id}
        # deleting another chat must not close the one that is open
        keep = None if chat_id == current_id else current_id
        return (*sidebar(chats, keep, hist), hist)

    async def refresh_sidebar(self, access_token, chat_id):
        # outputs: chats_state
        return await self._api.list_chats(access_token)

    # --------------------------------------------------------------- messages
    # outputs: chatbot, msg, current_chat, histories
    async def send(
        self, access_token, message: str, history: list, chat_id: str | None, hist: dict
    ):
        message = (message or "").strip()
        if not message:
            yield history, "", chat_id, hist
            return

        history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""},
        ]
        yield history, "", chat_id, hist

        try:
            async for event_name, data in self._api.stream_message(
                message, chat_id, token=access_token
            ):
                if event_name == "chat":
                    chat_id = data["chat_id"]
                elif event_name == "token":
                    history[-1]["content"] += data["text"]
                    yield history, "", chat_id, hist
                elif event_name == "tool":
                    self._apply_tool_event(history, data)
                    yield history, "", chat_id, hist
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

    @staticmethod
    def _apply_tool_event(history: list, data: dict) -> None:
        """Show a tool call as a collapsible block above the streaming answer."""
        name = data["name"]
        if data["status"] == "started":
            tool_block = {
                "role": "assistant",
                "content": "",
                "metadata": {"title": name, "status": "pending"},
            }
            history.insert(-1, tool_block)  # keep the answer bubble last
            return
        for item in history:
            meta = item.get("metadata")
            if (
                isinstance(meta, dict)
                and meta.get("status") == "pending"
                and meta.get("title") == name
            ):
                meta["status"] = "done"
                break

    # ------------------------------------------------------------------ voice
    # outputs: voice_frame, voice_panel, voice_timer
    async def voice_open(self, access_token: str | None, chat_id: str | None):
        if not access_token:
            gr.Warning("Log in first")
            return noops(3)

        if not chat_id:
            chat = await self._api.create_chat(access_token, "Voice chat")
            chat_id = chat["id"]
        # fragment (#...) is never sent to a server or written to logs
        fragment = urlencode({"token": access_token, "chat_id": chat_id})
        src = escape(f"{self._api_url}/voice-ui/#{fragment}")
        frame = (
            f'<iframe src="{src}" allow="microphone; autoplay" '
            'style="width:100%;height:56px;border:0;border-radius:12px">'
            "</iframe>"
        )
        return frame, gr.Column(visible=True), gr.Timer(active=True)

    def voice_close(self):
        # removing the iframe destroys the page -> LiveKit room disconnects
        return "", gr.Column(visible=False), gr.Timer(active=False)

    # outputs: chatbot
    async def poll_chat(self, access_token, chat_id, history):
        # a chat without an id yet (brand-new voice chat) has nothing to poll
        if not access_token or not chat_id:
            return NOOP()
        try:
            messages = await self._api.get_messages(access_token, chat_id)
        except ApiError:
            return NOOP()
        if len(messages) == len(history):
            return NOOP()
        return messages
