from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from nodes import (
    intent_classification,
    get_relevant_documents,
    ticket_node,
)
from state import MyState

config = {"configurable": {"thread_id": "Conversation_1"}}
builder = StateGraph(MyState)
builder.add_node("intent_classifier", intent_classification)
builder.add_node("get_relevant_documents", get_relevant_documents)
builder.add_node("ticket_node", ticket_node)
builder.add_edge(START, "intent_classifier")
builder.add_edge("intent_classifier", "get_relevant_documents")
builder.add_edge("get_relevant_documents", "ticket_node")
builder.add_edge("ticket_node", END)

compiled_graph = builder.compile(checkpointer=InMemorySaver())

inputs = {
    "messages": [
        {
            "role": "user",
            "content": "I am facing issue while connecting to VPN, please create a ticket for me.",
        }
    ],
    "user_id": "Teja",
}

# 1. Pass your actual `inputs` dictionary and use version="v2"
result = compiled_graph.invoke(inputs, config=config, version="v2")

while result.interrupts:
    # 1. Extract the prompt string passed to interrupt(...)
    prompt_message = result.interrupts[0].value
    
    # 2. Ask the human in the terminal (or get input from your UI/API)
    human_response = input(f"\n[Agent]: {prompt_message}\n[You]: ")
    
    # 3. Resume the graph by passing the string to Command(resume=...)
    result = compiled_graph.invoke(Command(resume=human_response), config=config, version="v2")

print("\nFinal response:", result.value["messages"][-1].content)

