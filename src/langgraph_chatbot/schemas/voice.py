import uuid

from pydantic import BaseModel


class VoiceTokenIn(BaseModel):
    chat_id: uuid.UUID | None = None  # None = start a new chat


class VoiceTokenOut(BaseModel):
    chat_id: uuid.UUID
    url: str
    token: str
