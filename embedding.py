from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single string into an embedding vector.
        """
        embedding = self.model.encode(text)
        # ChromaDB expects embeddings as lists of floats
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a batch of strings into corresponding embedding vectors.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
