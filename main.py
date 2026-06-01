import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from services.embedder import EmbeddingService
from services.vector_store import VectorStoreService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Cognitive RAG & Semantic Cache Service")

# Setup CORS for React frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to your portfolio domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
embedder = EmbeddingService()
vector_store = VectorStoreService()

# In-memory document counter to assign unique integer IDs for Qdrant points
document_counter = 0

class QueryRequest(BaseModel):
    text: str
    threshold: float = 0.90

class DocumentRequest(BaseModel):
    content: str
    metadata: dict = {}

# Simple Mock LLM generator for offline testing if GEMINI_API_KEY is not set
def mock_generate_answer(query: str, context: str) -> str:
    return f"[Mock LLM Response]\nBased on the retrieved context:\n\"{context}\"\n\nHere is the answer to your query: \"{query}\"."

def generate_llm_response(query: str, context: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in env. Falling back to Mock Mode.")
        return mock_generate_answer(query, context)
        
    try:
        # Simple client fallback or REST call to Gemini API to avoid dependency issues
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        You are a helpful assistant. Use the following piece of context to answer the user's question.
        If the context is empty or irrelevant, answer based on your knowledge but note that no relevant context was found.
        
        Context:
        {context}
        
        Question:
        {query}
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}. Falling back to Mock Mode.")
        return mock_generate_answer(query, context)


@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Main entrypoint for queries. Executes semantic cache lookup and falls back to RAG retrieval.
    """
    query_text = request.text
    threshold = request.threshold
    
    # 1. Convert incoming query text to vector embedding
    query_vector = embedder.get_embedding(query_text)
    
    # 2. Check the semantic cache first
    cache_results = vector_store.search_cache(query_vector, threshold=threshold)
    
    if cache_results:
        # Cache Hit!
        logger.info("Semantic cache HIT!")
        cached_payload = cache_results[0]["payload"]
        return {
            "source": "cache",
            "answer": cached_payload["answer"],
            "score": cache_results[0]["score"],
            "trace": ["Cache Search: HIT"]
        }
        
    # Cache Miss - retrieve from KB and generate response
    logger.info("Semantic cache MISS. Retrieving context from Knowledge Base...")
    kb_results = vector_store.search_knowledge(query_vector, limit=3)
    
    # Concatenate retrieved text chunks
    context_chunks = [r["payload"]["content"] for r in kb_results]
    context = "\n---\n".join(context_chunks) if context_chunks else "No relevant context found."
    
    # Call the LLM (or mock)
    answer = generate_llm_response(query_text, context)
    
    # Save query + answer vector to the cache for future queries
    cache_id = str(uuid.uuid4())
    vector_store.upsert_cache(
        cache_id=cache_id,
        vector=query_vector,
        payload={"query": query_text, "answer": answer}
    )
    
    return {
        "source": "llm",
        "answer": answer,
        "score": None,
        "trace": [
            "Cache Search: MISS",
            f"KB Retrieval: SUCCESS ({len(kb_results)} chunks found)",
            "LLM Generation: SUCCESS"
        ]
    }


@app.post("/knowledge")
async def add_knowledge(doc: DocumentRequest):
    """
    Endpoint to populate the knowledge base with information chunks.
    """
    global document_counter
    document_counter += 1
    
    text = doc.content
    vector = embedder.get_embedding(text)
    
    # Save to Qdrant Knowledge Base
    payload = {"content": text, "metadata": doc.metadata}
    vector_store.upsert_knowledge(document_id=document_counter, vector=vector, payload=payload)
    
    return {"message": "Document added to knowledge base", "id": document_counter}


@app.post("/cache/clear")
async def clear_cache():
    """
    Endpoint to purge the semantic cache.
    """
    vector_store.clear_cache()
    return {"message": "Semantic cache cleared successfully"}


@app.get("/health")
async def health_check():
    """
    Health check.
    """
    return {"status": "ok"}
