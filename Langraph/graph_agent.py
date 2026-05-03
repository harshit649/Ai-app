# graph_agent.py
import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END, add_messages

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
# tool_map = {"calculate": calculate_function}
# So we can find the right tool by name

# ===== LLM WITH TOOLS =====
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)
# bind_tools = "hey LLM, these tools exist. You can call them."

# ===== NODE 1: CHATBOT =====
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ===== NODE 2: TOOL EXECUTOR =====
def tool_node(state: State):
    # Get the last message (which has tool calls)
    last_message = state["messages"][-1]
    
    results = []
    for tool_call in last_message.tool_calls:
        # Find the right tool
        tool_fn = tool_map[tool_call["name"]]
        
        # Run it
        result = tool_fn.invoke(tool_call["args"])
        
        # Create a ToolMessage with the result
        results.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        )
    
    return {"messages": results}

# ===== CONDITIONAL EDGE: DOES LLM WANT TO USE TOOL? =====
def should_use_tool(state: State):
    last_message = state["messages"][-1]
    
    # If LLM's response has tool_calls → go to tool_node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise → done, go to END
    return "end"

# ===== BUILD THE GRAPH =====
graph = StateGraph(State)

# Add nodes
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "chatbot")         # start → chatbot
graph.add_conditional_edges(              # chatbot → tools OR end
    "chatbot",
    should_use_tool,
    {"tools": "tools", "end": END}
)
graph.add_edge("tools", "chatbot")       # tools → back to chatbot

# Compile
app = graph.compile()

# ===== TEST =====
# Question that needs tool
print("=== Question 1: Math ===")
result = app.invoke({"messages": [("user", "What is 25 * 48?")]})
for msg in result["messages"]:
    if hasattr(msg, 'content') and msg.content:
        role = msg.__class__.__name__
        print(f"[{role}]: {msg.content}")
print()

# Question that does NOT need tool
print("=== Question 2: General ===")
result = app.invoke({"messages": [("user", "What is Python?")]})
for msg in result["messages"]:
    if hasattr(msg, 'content') and msg.content:
        role = msg.__class__.__name__
        print(f"[{role}]: {msg.content}")