from dataclasses import dataclass
import uuid

# from langchain_community.tools.tavily_search import TavilySearchResults


@dataclass
class ContextSchema:
    user_id: uuid.UUID

    # tavily_tool: TavilySearchResults
