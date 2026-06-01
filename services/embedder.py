import os
import logging
import torch
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Lazy imports inside functions to prevent importing optimum/transformers unless ONNX is selected
# from optimum.onnxruntime import ORTModelForFeatureExtraction
# from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedding service container. Engines are loaded lazily.
        """
        self.model_name = model_name
        self.pytorch_model = None
        self.onnx_model = None
        self.onnx_tokenizer = None
        self.google_client = None

    def _get_pytorch_model(self):
        if self.pytorch_model is None:
            logger.info("Initializing local PyTorch model (SentenceTransformers)...")
            self.pytorch_model = SentenceTransformer(self.model_name)
            logger.info("Local PyTorch model loaded successfully.")
        return self.pytorch_model

    def _get_onnx_model(self):
        if self.onnx_model is None:
            logger.info("Initializing local ONNX model (Optimum)...")
            # Lazy import to avoid loading transformers at container startup
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            
            self.onnx_tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{self.model_name}")
            self.onnx_model = ORTModelForFeatureExtraction.from_pretrained(
                f"sentence-transformers/{self.model_name}", 
                export=True
            )
            logger.info("Local ONNX model loaded successfully.")
        return self.onnx_model, self.onnx_tokenizer

    def _get_google_client(self):
        if self.google_client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set. Cannot run Google Cloud embeddings.")
            logger.info("Initializing Google GenAI client for Cloud Embeddings...")
            self.google_client = genai.Client(api_key=api_key)
        return self.google_client

    def _encode_onnx(self, text: str) -> list[float]:
        model, tokenizer = self._get_onnx_model()
        
        # Tokenize and execute model
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Perform Mean Pooling to get a single sentence representation
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        # Normalize embedding to unit length (L2 norm)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding[0].tolist()

    def _encode_google(self, text: str) -> list[float]:
        client = self._get_google_client()
        # Query text-embedding-004 model, truncating the output to 384 dimensions to match Qdrant schema
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=384
            )
        )
        return response.embeddings[0].values

    def get_embedding(self, text: str, engine: str = "pytorch") -> list[float]:
        """
        Convert a single string into a 384-dimensional vector using PyTorch, ONNX, or Google API.
        """
        engine_key = engine.lower().strip()
        
        if engine_key == "onnx":
            return self._encode_onnx(text)
        elif engine_key == "google":
            return self._encode_google(text)
        else:
            # Default to PyTorch
            model = self._get_pytorch_model()
            embedding = model.encode(text)
            return embedding.tolist()
