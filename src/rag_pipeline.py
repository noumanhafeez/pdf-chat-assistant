from utils.logging_integeration import get_logger

logger = get_logger("rag_pipeline", "logs/rag_pipeline.log")


class RAGService:

    def __init__(self, vector_store, embedding_service, llm):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm = llm

    def answer_question(self, question: str):

        logger.info(f"Received query: {question}")

        query_embedding = self.embedding_service.get_embeddings([question])

        docs = self.vector_store.search(query_embedding)

        context = "\n".join(docs)

        answer = self.llm.generate_answer(question, context)

        return answer