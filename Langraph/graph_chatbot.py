import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, add_messages

load_dotenv()

# step 1 define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# step 2 define llm
llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature=0.0)

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# build the graph

graph = StateGraph(State)

graph.add_node("chatbot", chatbot)

#add edges
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app=graph.compile()

#step 5 test
result = app.invoke({
    "messages": [("user", "What is Python?")]
})

#print the conversation
for msg in result["messages"]:
    role = "user" if msg.type == "human" else "AI"
    print(f"{role}: {msg.content}")
