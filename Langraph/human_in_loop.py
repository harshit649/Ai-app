# human_in_loop.py
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ===== TOOLS =====
@tool
def delete_records(older_than_years: int) -> str:
    """Deletes database records older than specified years."""
    return f"Deleted all records older than {older_than_years} years"

@tool
def search_web(query: str) -> str:
    """Searches the web for information."""
    return f"Search results for: {query}"

tools = [delete_records, search_web]
tool_map = {t.name: t for t in tools}

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ===== NODES =====
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: State):
    last_message = state["messages"][-1]
    results = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        
        # DANGEROUS tools need human approval
        dangerous_tools = ["delete_records"]
        
        if tool_name in dangerous_tools:
            # PAUSE HERE — ask human
            human_response = interrupt(
                f"Agent wants to run '{tool_name}' with args: {tool_call['args']}. "
                f"Approve? (yes/no)"
            )
            
            if human_response.lower() != "yes":
                results.append(
                    ToolMessage(
                        content="Action denied by human. Not executed.",
                        tool_call_id=tool_call["id"]
                    )
                )
                continue
        
        # Safe tools or approved dangerous tools — run normally
        tool_fn = tool_map[tool_name]
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

# ===== BUILD GRAPH =====
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", should_use_tool, {"tools": "tools", "end": END})
graph.add_edge("tools", "chatbot")

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "test_001"}}

# ===== TEST 1: Safe tool (no approval needed) =====
print("--- Test 1: Safe tool ---")
result = app.invoke(
    {"messages": [("user", "Search the web for AI news")]},
    config=config
)
print(result["messages"][-1].content)

# ===== TEST 2: Dangerous tool (needs approval) =====
print("\n--- Test 2: Dangerous tool ---")
config2 = {"configurable": {"thread_id": "test_002"}}
result = app.invoke(
    {"messages": [("user", "Delete all records older than 5 years")]},
    config=config2
)
# Graph PAUSES here at interrupt
# Check if graph is waiting for input
print("Graph paused. Waiting for human approval...")

# Resume with human approval
result = app.invoke(
    Command(resume="yes"),   # human says "yes"
    config=config2
)
print(result["messages"][-1].content)

# streaming_agent.py
# (after building and compiling your graph as before)

# config = {"configurable": {"thread_id": "stream_001"}}

# # stream_mode="messages" gives you token-by-token
# for chunk, metadata in app.stream(
#     {"messages": [("user", "Explain what RAG is in 3 sentences")]},
#     config=config,
#     stream_mode="messages"
# ):
#     # Only print AI's response tokens (not tool calls etc)
#     if hasattr(chunk, 'content') and chunk.content:
#         print(chunk.content, end="", flush=True)

# print()  # newline at end