import uuid
from dataclasses import dataclass

# from langchain_community.tools.tavily_search import TavilySearchResults


@dataclass
class ContextSchema:
    user_id: uuid.UUID
    chat_id: uuid.UUID

    # tavily_tool: TavilySearchResults
