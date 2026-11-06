import uuid

from pydantic import BaseModel, ConfigDict, Field


class ChatRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str


class ChatCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=255)


class ChatUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class MessageIn(BaseModel):
    message: str
    chat_id: uuid.UUID | None = None


class MessageRead(BaseModel):
    role: str
    content: str
