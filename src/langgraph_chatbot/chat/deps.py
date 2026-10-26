from typing import Annotated

from fastapi import Depends

from langgraph_chatbot.chat.repository import ChatRepository
from langgraph_chatbot.chat.service import ChatService
from langgraph_chatbot.deps import GraphDep, SessionDep


def get_chat_repo(session: SessionDep) -> ChatRepository:
    return ChatRepository(session)


def get_chat_service(
    session: SessionDep, chats: ChatRepoDep, graph: GraphDep
) -> ChatService:
    return ChatService(session, chats, graph)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
ChatRepoDep = Annotated[ChatRepository, Depends(get_chat_repo)]
