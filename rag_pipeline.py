import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Clients
client_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# Vector DB
client_vectordb = chromadb.PersistentClient(path="vector_db")
collection = client_vectordb.get_collection("movie_reviews")


def rag_answer(question):
    # Embed question
    query_embedding = model_embed.encode(question).tolist()

    # Retrieve top 5 chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # Handle no results
    docs = results["documents"][0]
    if len(docs) == 0:
        return "No relevant documents found.", []

    # Combine context
    context = "\n\n".join(docs)

    # RAG Prompt
    prompt = f"""
Use ONLY the following movie review evidence to answer the question.
If evidence is insufficient, say "Not enough information in reviews."

Evidence:
{context}

Question: {question}
"""

    # Query GROQ LLM
    response = client_llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()

    return answer, docs
