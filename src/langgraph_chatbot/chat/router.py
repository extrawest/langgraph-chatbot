import uuid
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent

from langgraph_chatbot.chat.deps import ChatServiceDep
from langgraph_chatbot.chat.utils import make_title
from langgraph_chatbot.deps import CurrentUser
from langgraph_chatbot.models.chat import Chat
from langgraph_chatbot.schemas.chat import (
    ChatCreate,
    ChatRead,
    ChatUpdate,
    MessageIn,
    MessageRead,
)

router = APIRouter(prefix="/chats")


@router.get("", response_model=list[ChatRead])
async def list_user_chats(current_user: CurrentUser, service: ChatServiceDep):
    chats = await service.list_chats(current_user.id)

    return chats


@router.post("", response_model=ChatRead, status_code=201)
async def create_chat(
    data: ChatCreate, current_user: CurrentUser, service: ChatServiceDep
):
    return await service.create_chat(current_user.id, data.title)


@router.get("/{chat_id}", response_model=ChatRead)
async def get_chat(
    chat_id: uuid.UUID, current_user: CurrentUser, service: ChatServiceDep
):
    chat = await service.get_chat(chat_id, current_user.id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.get("/{chat_id}/messages", response_model=list[MessageRead])
async def get_chat_messages(
    chat_id: uuid.UUID, current_user: CurrentUser, service: ChatServiceDep
):
    messages = await service.get_messages(chat_id, current_user.id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return messages


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


async def get_chat_for_message(
    data: MessageIn, current_user: CurrentUser, service: ChatServiceDep
) -> Chat:
    """Runs before streaming starts, so a missing chat is a real 404, not a broken stream."""

    chat = await service.get_or_create_chat(
        chat_id=data.chat_id,
        user_id=current_user.id,
        title=make_title(data.message),
    )
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


ChatForMessage = Annotated[Chat, Depends(get_chat_for_message)]


@router.post("/messages", response_class=EventSourceResponse)
async def send_message(
    data: MessageIn,
    chat: ChatForMessage,
    current_user: CurrentUser,
    service: ChatServiceDep,
) -> AsyncIterable[ServerSentEvent]:
    async for event in service.stream_reply(chat, current_user.id, data.message):
        yield ServerSentEvent(data=event, event=event.event_name)
