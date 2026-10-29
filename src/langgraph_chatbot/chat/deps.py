from typing import Annotated

from fastapi import Depends, Request

from langgraph_chatbot.chat.repositories.messages_store import MessagesStore
from langgraph_chatbot.chat.repositories.repository import ChatRepository
from langgraph_chatbot.chat.service import ChatService
from langgraph_chatbot.deps import GraphDep, SessionDep


def get_chat_repo(session: SessionDep) -> ChatRepository:
    return ChatRepository(session)


def get_chat_service(
    session: SessionDep,
    chats: ChatRepoDep,
    graph: GraphDep,
    messages_store: MessagesStoreDep,
) -> ChatService:
    return ChatService(session, chats, graph, messages_store)


def get_messages_store(request: Request) -> MessagesStore:
    return request.app.state.messages_store


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
ChatRepoDep = Annotated[ChatRepository, Depends(get_chat_repo)]
MessagesStoreDep = Annotated[MessagesStore, Depends(get_messages_store)]
