import uuid
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_name: ClassVar[str]


class ChatEvent(StreamEvent):
    event_name: ClassVar[str] = "chat"

    chat_id: uuid.UUID
    title: str


class TokenEvent(StreamEvent):
    event_name: ClassVar[str] = "token"

    text: str


class ToolEvent(StreamEvent):
    event_name: ClassVar[str] = "tool"

    name: str
    status: str  # "started" | "finished"


class ErrorEvent(StreamEvent):
    event_name: ClassVar[str] = "error"

    detail: str


class DoneEvent(StreamEvent):
    event_name: ClassVar[str] = "done"
