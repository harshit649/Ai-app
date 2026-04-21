# agent.py (full version)
import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
# from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 1. Define tools (the agent's hands)
search_tool = DuckDuckGoSearchResults(max_results=3)
tools = [search_tool]

# 2. Define the brain (LLM)
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

# python -m pip install langchain-google-genai

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 3. Create the agent (brain + tools combined)
agent = create_react_agent(llm, tools)
# agent = create_agent(llm, tools)

# 4. Run it!
# result = agent.invoke({
#     "messages": [
#         ("user", "What are the top 3 AI frameworks in 2026?")
#     ]
# })

# result = agent.invoke({
#     "messages": [("user", "What is 2 + 2?")]
# })

result = agent.invoke({
    "messages": [
        ("user", "Compare the salary of AI engineers in India vs USA in 2026")
    ]
})

# 5. Print the full conversation — see agent THINKING
for msg in result["messages"]:
    role = msg.__class__.__name__
    
    # Show tool calls if any
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"\n[{role} — DECIDED TO USE TOOL]")
        for tc in msg.tool_calls:
            print(f"  Tool: {tc['name']}")
            print(f"  Input: {tc['args']}")
    elif hasattr(msg, 'content') and msg.content:
        print(f"\n[{role}]")
        print(msg.content)
    print("-" * 40)