from sentence_transformers import SentenceTransformer
import numpy as np
from utils.logging_integeration import get_logger


logger = get_logger("embeddings", "logs/embeddings.log")

class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embeddings(self, texts: list[str]):
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(texts)
        return np.array(embeddings)