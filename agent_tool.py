# agent.py
import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Create the search tool — agent can use this to search the web
search_tool = DuckDuckGoSearchResults(max_results=3)

# Test the tool manually first
print(search_tool.invoke("will the sensex tomorrow will open in green?"))