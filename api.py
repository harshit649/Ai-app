# # api.py
# from fastapi import FastAPI

# app = FastAPI(title="AI Agent API")

# @app.get("/health")
# async def health():
#     return {"status": "ok"}

# # Run with: python -m uvicorn api:app --reload


# api.py (full version)
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ===== AGENT SETUP (runs once at startup) =====
search_tool = DuckDuckGoSearchResults(max_results=3)

@tool
def calculate(expression: str) -> str:
    """Calculates a math expression. Use for any math."""
    try:
        return f"Result: {eval(expression)}"
    except:
        return "Error: could not calculate"

tools = [search_tool, calculate]
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
memory = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=memory)

# ===== API SETUP =====
app = FastAPI(
    title="AI Agent API",
    description="Production AI Agent with memory and tools"
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

# ===== ENDPOINTS =====
@app.get("/health")
async def health():
    return {"status": "ok", "agent": "running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.session_id}}
    
    result = agent.invoke(
        {"messages": [("user", request.message)]},
        config=config
    )
    
    ai_reply = result["messages"][-1].content
    
    return ChatResponse(
        response=ai_reply,
        session_id=request.session_id
    )