from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from langgraph_chatbot.auth.security import create_access_token
from langgraph_chatbot.auth.service import authenticate_user
from langgraph_chatbot.config import get_settings
from langgraph_chatbot.deps import get_current_active_user
from langgraph_chatbot.schemas.user import UserRead

router = APIRouter(prefix="users")


@router.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await authenticate_user("fake_db", form_data.username, form_data.password)
    settings = get_settings()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    encoded_jwt = create_access_token(user, settings.jwt_secret, settings.jwt_algorithm)

    return {"access_token": encoded_jwt, "token_type": "bearer"}


@router.get("/users/me")
async def read_users_me(
    current_user: Annotated[UserRead, Depends[get_current_active_user]],
):
    return current_user
