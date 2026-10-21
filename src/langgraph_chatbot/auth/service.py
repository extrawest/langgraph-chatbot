from langgraph_chatbot.auth.security import verify_password
from langgraph_chatbot.deps import get_user


async def authenticate_user(db, username: str, password: str):
    user = await get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user
