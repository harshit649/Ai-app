import os
from dotenv import load_dotenv
from groq import Groq
import numpy as np

load_dotenv()

# we r using diff api for embeddings as groq doesnt support embeddings

from sentence_transformers import SentenceTransformer

# This model runs LOCALLY on your machine — no API key needed, completely free
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert sentences to embeddings
sentences = [
    "I love dogs",
    "Puppies are amazing pets",
    "The stock market crashed today",
    "Python is a programming language",
    "I enjoy coding in Python"
]

embeddings = model.encode(sentences)
print("embeddings", embeddings)
print("embeddings 0", embeddings[0])

print("Shape of one embedding:", embeddings[0].shape)
# Will print something like (384,) → 384 numbers per sentence

# Now measure similarity between sentences
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare pairs
print("\nSimilarity scores (1.0 = identical, 0.0 = completely different):")
print(f"'I love dogs' vs 'Puppies are amazing': {cosine_similarity(embeddings[0], embeddings[1]):.3f}")
print(f"'I love dogs' vs 'Stock market crashed': {cosine_similarity(embeddings[0], embeddings[2]):.3f}")
print(f"'Python language' vs 'I enjoy coding in Python': {cosine_similarity(embeddings[3], embeddings[4]):.3f}")
