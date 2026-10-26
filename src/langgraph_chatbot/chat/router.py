import uuid
from collections.abc import AsyncIterable

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from langgraph_chatbot.chat.deps import ChatServiceDep
from langgraph_chatbot.chat.events import StreamEvent, TokenEvent
from langgraph_chatbot.deps import CurrentUser
from langgraph_chatbot.schemas.chat import ChatRead, ChatUpdate, MessageIn

router = APIRouter(prefix="/chats")


@router.get("", response_model=list[ChatRead])
async def list_user_chats(current_user: CurrentUser, service: ChatServiceDep):
    chats = await service.list_chats(current_user.id)

    return chats


@router.get("/{chat_id}", response_model=ChatRead)
async def get_chat(current_user: CurrentUser, service: ChatServiceDep, chat_id: str):
    chat = await service.get_chat(chat_id, current_user.id)

    response = ChatRead.model_validate(chat)
    return response


@router.patch("/{chat_id}", response_model=ChatRead)
async def rename_chat(
    chat_id: uuid.UUID,
    data: ChatUpdate,
    current_user: CurrentUser,
    service: ChatServiceDep,
):
    chat = await service.rename_chat(chat_id, current_user.id, data.title)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("/messages", response_class=EventSourceResponse)
async def chat(
    current_user: CurrentUser,
    service: ChatServiceDep,
    data: MessageIn,
) -> AsyncIterable[ServerSentEvent]:
    if not data.chat_id:
        chat = await service.create_chat(current_user.id)
    else:
        chat = await service.get_chat(current_user.id)

    async for event in service.stream_reply(chat.id, current_user.id, data.message):
        yield ServerSentEvent(data=event, event=event.event_name)
