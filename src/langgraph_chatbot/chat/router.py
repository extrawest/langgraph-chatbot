import uuid

from fastapi import APIRouter, HTTPException

from langgraph_chatbot.chat.deps import ChatServiceDep
from langgraph_chatbot.deps import AuthServiceDep, CurrentUser
from langgraph_chatbot.schemas.chat import ChatRead, ChatUpdate, UserChats
from langgraph_chatbot.schemas.user import UserCreate, UserRead
from langgraph_chatbot.users.repository import EmailAlreadyExists

router = APIRouter(prefix="/chats")


@router.get("/", response_model=UserChats)
async def list_user_chats(current_user: CurrentUser, service: ChatServiceDep):
    chats = await service.list_chats(current_user.id)

    response = UserChats.model_validate(chats)
    return response


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


@router.post("/register", response_model=UserRead, status_code=201)
async def register(data: UserCreate, auth_service: AuthServiceDep):
    try:
        user = await auth_service.register(data.email, data.password)
        response = UserRead.model_validate(user)
        return response
    except EmailAlreadyExists:
        raise HTTPException(
            status_code=409,
            detail="Email already registered",
        )
