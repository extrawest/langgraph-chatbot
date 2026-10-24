from langgraph.graph import MessagesState


class AppState(MessagesState):
    user_id: str
