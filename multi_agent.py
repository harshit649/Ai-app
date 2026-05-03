# multi_agent.py
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process

load_dotenv()

# Define the LLM
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ===== AGENT 1: RESEARCHER =====
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find accurate, detailed information on any topic",
    backstory="""You are an expert researcher. You find credible 
    information and present it clearly with key facts and numbers.""",
    llm=llm,
    verbose=True    # shows thinking process
)

# ===== AGENT 2: WRITER =====
writer = Agent(
    role="Content Writer",
    goal="Write clear, engaging content from research",
    backstory="""You are a skilled writer who turns complex research 
    into easy-to-understand content. You write concisely.""",
    llm=llm,
    verbose=True
)

# ===== TASK 1: RESEARCH =====
research_task = Task(
    description="""Research the top 3 use cases of AI agents in 2026. 
    For each use case include: what it does, which companies use it, 
    and why it matters.""",
    expected_output="A detailed list of 3 use cases with facts.",
    agent=researcher
)

# ===== TASK 2: WRITE (uses research output) =====
writing_task = Task(
    description="""Using the research provided, write a short 
    engaging blog post (200 words max) about AI agent use cases. 
    Make it interesting for a non-technical audience.""",
    expected_output="A 200-word blog post with an engaging title.",
    agent=writer,
    context=[research_task]    # ← gets output of research task
)

# ===== CREW: COMBINE AGENTS + TASKS =====
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.hierarchical,
    manager_llm=llm,
    verbose=True
)

# ===== RUN IT =====
result = crew.kickoff()
print("\n\n===== FINAL OUTPUT =====")
print(result)