import faiss
from utils.logging_integeration import get_logger

logger = get_logger("faiss_database", "logs/faiss_database.log")

class VectorStore:

    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

    def add(self, embeddings, chunks):
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks to vector DB")

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding, k)
        results = [self.text_chunks[i] for i in indices[0]]
        return results