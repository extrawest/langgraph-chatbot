from langgraph.graph import MessagesState


class AppState(MessagesState):
    # chat history loaded from Qdrant, kept out of `messages` on purpose: the
    # add_messages reducer only appends, so history placed there would end up
    # after the question it precedes. Plain dicts (not BaseMessage) on purpose:
    # stream_mode="messages" forwards every BaseMessage a node returns, so history
    # as messages would leak into the stream (and get spoken aloud in voice mode)
    history: list[dict]
    # long-term facts about the user recalled from Mem0 (cross-chat)
    memories: list[str]
