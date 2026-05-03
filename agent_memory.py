# memory_agent.py
import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Tools
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

# THE KEY LINE — add memory
memory = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=memory)

# thread_id = unique ID for this conversation
# Same thread_id = same conversation = agent remembers
# Different thread_id = new conversation = agent forgets
config = {"configurable": {"thread_id": "harshit_session_1"}}

# Call 1
result1 = agent.invoke(
    {"messages": [("user", "My name is Harshit and I earn 11 LPA")]},
    config=config
)
print(result1["messages"][-1].content)

# Call 2 — agent should remember
result2 = agent.invoke(
    {"messages": [("user", "What is my name and salary?")]},
    config=config
)
print(result2["messages"][-1].content)

# Call 3 — agent should use tool + memory together
result3 = agent.invoke(
    {"messages": [("user", "Calculate my monthly salary after 30% tax")]},
    # config={"configurable": {"thread_id": "harshit_session_2"}}
    config = config
)
print(result3["messages"][-1].content)