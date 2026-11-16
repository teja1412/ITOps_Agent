from langgraph.graph import MessagesState


class MyState(MessagesState):
    user_id: str
    intent : str