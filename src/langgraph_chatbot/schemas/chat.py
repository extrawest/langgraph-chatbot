import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

# stripped before the length check, so "   " is rejected instead of saved as ""
ChatTitle = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
]


class ChatRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str


class ChatCreate(BaseModel):
    title: ChatTitle = "New chat"


class ChatUpdate(BaseModel):
    title: ChatTitle


class MessageIn(BaseModel):
    message: str
    chat_id: uuid.UUID | None = None


class MessageRead(BaseModel):
    role: str
    content: str
