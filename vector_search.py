# vector_search.py
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Imagine these are paragraphs from your documentation
documents = [
    "To reset your password, go to Settings and click Forgot Password.",
    "Our pricing plans start from $10 per month for the basic plan.",
    "You can export your data as CSV from the Reports section.",
    "To contact support, email us at support@company.com",
    "The API rate limit is 1000 requests per hour for free accounts.",
]

# Embed all documents — store these in vector DB in real life
doc_embeddings = model.encode(documents)

def search(query, top_k=2):
    # Embed the query
    query_embedding = model.encode([query])[0]
    print("query_embedding", query_embedding)
    # Calculate similarity with every document
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        score = np.dot(query_embedding, doc_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
        )
        similarities.append((score, documents[i]))
    
    # Sort by similarity score
    similarities.sort(reverse=True)
    
    print(f"\nQuery: '{query}'")
    print("query[0]", query[0])
    print("Top results:")
    for score, doc in similarities[:top_k]:
        print(f"  Score {score:.3f} → {doc}")

# Test your search engine
search("How do I change my password?")
search("What does the API allow me to do?")
search("I need help, who do I contact?")