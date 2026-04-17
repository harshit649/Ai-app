# rag_system.py
# building your first rag system
import os
from dotenv import load_dotenv
from groq import Groq

# LangChain components
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ========== PHASE 1: INDEXING ==========

# Step 1 - Load your document
loader = TextLoader("my_knowledge.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document")

# Step 2 - Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # max characters per chunk
    chunk_overlap=50      # overlap between chunks
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Step 3 - Create embeddings (free, runs locally)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Step 4 - Store in ChromaDB (free local vector database)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
print("Stored in vector database")

# ========== PHASE 2: QUERYING ==========

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(question):
    # Step 1 - Find relevant chunks
    relevant_chunks = vectorstore.similarity_search(question, k=3)
    
    # Step 2 - Build context from chunks
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # Step 3 - Send to LLM with context
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,    # precise answers for RAG
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant.
                Answer ONLY using the context provided below.
                If answer is not in context say 'I don't have this information.'
                
                Context:
                """ + context
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    
    answer = response.choices[0].message.content
    
    # Show which chunks were used (great for debugging)
    print(f"\nQuestion: {question}")
    print(f"Chunks used: {[c.page_content[:50] for c in relevant_chunks]}")
    print(f"Answer: {answer}")
    print("-" * 50)

# Test it!
ask("How do I reset my password?")
ask("What is the refund policy?")
ask("Who is the CEO?")
ask("What is the price of FlowAPI?")
ask("What is the capital of India?")  # not in document — watch what happens