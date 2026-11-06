"""Chat list in the sidebar: one row per chat with rename and delete buttons.

Rendered with `gr.render`, so the rows are rebuilt whenever the chats or the
selected chat change. Rename / delete do not edit the row itself: the buttons only
fill and show the static edit panel (see app.py), which keeps the rows simple.
"""

import gradio as gr

from langgraph_chatbot.ui.handlers import UiHandlers

ROW_BUTTON = {"scale": 1, "min_width": 36}


def render_chat_list(
    *,
    h: UiHandlers,
    token: gr.State,
    chats_state: gr.State,
    current_chat: gr.State,
    histories: gr.State,
    sidebar_outputs: list,
    edit_outputs: list,
) -> None:
    @gr.render(inputs=[chats_state, current_chat])
    def _rows(chats: list[dict], selected: str | None):
        if not chats:
            gr.Markdown("*No chats yet*")
            return

        for chat in chats:
            chat_id_state = gr.State(chat["id"])
            with gr.Row(elem_classes="chat-row"):
                open_btn = gr.Button(
                    chat["title"],
                    variant="primary" if chat["id"] == selected else "secondary",
                    elem_classes="chat-open",
                    scale=6,
                    min_width=0,
                )
                # shown on row hover only (see CSS in app.py)
                rename_btn = gr.Button("✏️", elem_classes="row-action", **ROW_BUTTON)
                delete_btn = gr.Button("🗑", elem_classes="row-action", **ROW_BUTTON)

            open_btn.click(
                h.select,
                [token, chat_id_state, chats_state, histories],
                sidebar_outputs,
            )
            rename_btn.click(
                h.begin_edit,
                [gr.State("rename"), chat_id_state, chats_state],
                edit_outputs,
            )
            delete_btn.click(
                h.begin_edit,
                [gr.State("delete"), chat_id_state, chats_state],
                edit_outputs,
            )
