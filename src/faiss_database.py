import faiss
from utils.logging_integeration import get_logger

logger = get_logger("faiss_database", "logs/faiss_database.log")

class VectorStore:

    def __init__(self, dim=768):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

    def add(self, embeddings, chunks):
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks to vector DB")

    def search(self, query_embedding, top_k=3):
        import numpy as np

        if len(self.text_chunks) == 0:
            return []

        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for i in indices[0]:
            if 0 <= i < len(self.text_chunks):
                results.append(self.text_chunks[i])

        return results