import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", ":memory:")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
        logger.info(f"Connecting to Qdrant at: {qdrant_url}")
        
        if qdrant_url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif qdrant_url.startswith("http://") or qdrant_url.startswith("https://"):
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # Treat as local directory path for persistence (e.g. "./qdrant_db")
            self.client = QdrantClient(path=qdrant_url)
        
        # Define our collections
        self.kb_collection = "knowledge_base"
        self.cache_collection = "semantic_cache"
        self.vector_size = 384  # MiniLM dimension
        
        self._ensure_collections()

    def _ensure_collections(self):
        """
        Creates collections if they do not exist.
        """
        for name in [self.kb_collection, self.cache_collection]:
            if not self.client.collection_exists(collection_name=name):
                logger.info(f"Creating Qdrant collection: {name}")
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )

    def upsert_knowledge(self, document_id: int, vector: list[float], payload: dict):
        """
        Inserts document chunks into the knowledge base collection.
        """
        self.client.upsert(
            collection_name=self.kb_collection,
            points=[
                PointStruct(
                    id=document_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def search_knowledge(self, query_vector: list[float], limit: int = 3) -> list[dict]:
        """
        Retrieves relevant documents for RAG.
        """
        results = self.client.query_points(
            collection_name=self.kb_collection,
            query=query_vector,
            limit=limit
        )
        return [{"score": r.score, "payload": r.payload} for r in results.points]

    def upsert_cache(self, cache_id: str, vector: list[float], payload: dict):
        """
        Caches a query vector and its generated answer payload.
        """
        # Convert string UUID/ID to a compatible format or use raw string in Qdrant (which supports string UUIDs)
        self.client.upsert(
            collection_name=self.cache_collection,
            points=[
                PointStruct(
                    id=cache_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def search_cache(self, query_vector: list[float], threshold: float = 0.90) -> list[dict]:
        """
        Searches the cache for semantically similar queries.
        """
        results = self.client.query_points(
            collection_name=self.cache_collection,
            query=query_vector,
            limit=1
        )
        
        # Filter results by similarity score threshold
        valid_results = []
        for r in results.points:
            if r.score >= threshold:
                valid_results.append({"score": r.score, "payload": r.payload})
                
        return valid_results
        
    def clear_cache(self):
        """
        Clears all entries in the semantic cache collection.
        """
        logger.info("Clearing semantic cache collection.")
        self.client.delete_collection(collection_name=self.cache_collection)
        # Re-ensure it exists
        self._ensure_collections()
