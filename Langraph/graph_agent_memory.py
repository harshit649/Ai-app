# graph_agent_memory.py
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ===== STATE =====
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ===== TOOLS =====
@tool
def calculate(expression: str) -> str:
    """Calculates a math expression. Use for any math."""
    try:
        return f"Result: {eval(expression)}"
    except:
        return "Error: could not calculate"

tools = [calculate]
tool_map = {t.name: t for t in tools}

# ===== LLM =====
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ===== NODES (same as Exercise 2) =====
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: State):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = tool_map[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )
    return {"messages": results}

def should_use_tool(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# ===== BUILD GRAPH (same as Exercise 2) =====
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", should_use_tool, {"tools": "tools", "end": END})
graph.add_edge("tools", "chatbot")

# ===== NEW: ADD MEMORY =====
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
#                   ↑ THIS is the only change

# ===== TEST WITH MEMORY =====
config = {"configurable": {"thread_id": "harshit_001"}}

# Call 1: Tell it something
print("--- Call 1 ---")
result1 = app.invoke(
    {"messages": [("user", "My name is Harshit and I earn 30 LPA")]},
    config=config
)
print(result1["messages"][-1].content)

# Call 2: Ask if it remembers
print("\n--- Call 2 ---")
result2 = app.invoke(
    {"messages": [("user", "What is my name and salary?")]},
    config=config
)
print(result2["messages"][-1].content)

# Call 3: Use memory + tool together
print("\n--- Call 3 ---")
result3 = app.invoke(
    {"messages": [("user", "Calculate my monthly salary")]},
    config=config
)
print(result3["messages"][-1].content)

# Call 4: Different user — should NOT know Harshit
print("\n--- Call 4 (different user) ---")
config2 = {"configurable": {"thread_id": "priya_001"}}
result4 = app.invoke(
    {"messages": [("user", "What is my name?")]},
    config=config2
)
print(result4["messages"][-1].content)