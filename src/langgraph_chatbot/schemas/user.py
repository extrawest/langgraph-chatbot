from pydantic import BaseModel


class UserCreate(BaseModel):
    email: str
    username: str


class UserRead(BaseModel):
    email: str
    username: str
    is_active: bool
