from typing import Annotated

from fastapi import Depends

from langgraph_chatbot.chat.repository import ChatRepository
from langgraph_chatbot.chat.service import ChatService
from langgraph_chatbot.deps import SessionDep


def get_chat_repo(session: SessionDep) -> ChatRepository:
    return ChatRepository(session)


def get_chat_service(session: SessionDep, chats: ChatRepository) -> ChatService:
    return ChatService(session, chats)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
