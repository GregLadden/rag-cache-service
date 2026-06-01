from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the local SentenceTransformer model.
        """
        logger.info(f"Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")

    def get_embedding(self, text: str) -> list[float]:
        """
        Convert a single string into a 384-dimensional dense vector representation.
        """
        embedding = self.model.encode(text)
        return embedding.tolist()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of strings into a list of 384-dimensional dense vectors.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
