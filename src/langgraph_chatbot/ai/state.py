from langgraph.graph import MessagesState


class AppState(MessagesState):
    # raw text of the current user turn; load_history turns it into a HumanMessage
    user_input: str
