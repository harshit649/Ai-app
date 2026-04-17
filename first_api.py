# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# # Load API key from .env file safely
# load_dotenv()

# # Create the client — your connection to OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Make your first API call
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     temperature=0.7,
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant."
#         },
#         {
#             "role": "user",
#             "content": "Explain what an API is in 2 sentences."
#         }
#     ]
# )

# # Extract the actual text from the response
# answer = response.choices[0].message.content

# print(answer)

# gemini api key

# import os
# from dotenv import load_dotenv
# from google import genai

# load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents="Explain what an API is in 2 sentences."
# )

# print(response.text)


# grok api key

# import os
# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# response = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",   # free + powerful
#     temperature=0.7,
#     # messages=[
#     #     {"role": "system", "content": "Act as helpful ai assistant"},
#     #     {"role": "user", "content": "Write one creative opening line for a story."},
#     #     {"role": "user", "content": "My name is Harshit"},
#     #     {"role": "user", "content": "What is my name"},

#     # ]
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "My name is Harshit"},
#         {"role": "assistant", "content": "Hello Harshit! How can I help you?"},
#         {"role": "user", "content": "What is my name?"},
#     ]
# )

# print(response.choices[0].message.content)


# Anthropic (Claude)
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(auth_token=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    system="You are helpful",        # ← separate param, not inside messages
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)
print(response.content[0].text)      # ← different extraction