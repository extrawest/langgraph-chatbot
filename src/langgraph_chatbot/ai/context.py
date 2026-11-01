import uuid
from dataclasses import dataclass
from enum import Enum

# from langchain_community.tools.tavily_search import TavilySearchResults


class Mode(Enum):
    VOICE = "voice"
    TEXT = "text"


@dataclass
class ContextSchema:
    user_id: uuid.UUID
    chat_id: uuid.UUID

    mode: Mode = Mode.TEXT

    # tavily_tool: TavilySearchResults
