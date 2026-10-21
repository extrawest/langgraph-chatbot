from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from langgraph_chatbot.auth.security import create_access_token
from langgraph_chatbot.deps import AuthServiceDep, CurrentUser, SettingsDep
from langgraph_chatbot.schemas.user import UserCreate, UserRead

router = APIRouter(prefix="users")


@router.post("/token")
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    auth_service: AuthServiceDep,
    settings: SettingsDep,
):
    user = await auth_service.authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    encoded_jwt = create_access_token(user, settings.jwt_secret, settings.jwt_algorithm)

    return {"access_token": encoded_jwt, "token_type": "bearer"}


@router.get("/users/me")
async def read_users_me(current_user: CurrentUser):
    return current_user


@router.post("/register", response_model=UserRead, status_code=201)
async def register(data: UserCreate, service: AuthServiceDep):
    pass
