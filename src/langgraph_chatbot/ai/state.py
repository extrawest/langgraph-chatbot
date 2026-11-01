from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState


class AppState(MessagesState):
    # chat history loaded from Qdrant, kept out of `messages` on purpose: the
    # add_messages reducer only appends, so history placed there would end up
    # after the question it precedes
    history: list[BaseMessage]
    # long-term facts about the user recalled from Mem0 (cross-chat)
    memories: list[str]
