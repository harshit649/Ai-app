import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# Step A: Define the prompt template (the waiter's slip)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")   # {question} is a variable slot
])

# Step B: Define the LLM (the chef)
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Step C: Define output cleaner (the plating)
parser = StrOutputParser()

# Step D: Connect them with pipe |
chain = prompt | llm | parser

# Step E: Run it
answer = chain.invoke({"question": "What is Python?"})
print(answer)