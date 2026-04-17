import os

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ocr_service import OCRService
from src.embeddings import EmbeddingService
from src.faiss_database import VectorStore
from src.rag_pipeline import RAGService
from src.hugging_face import HuggingFaceLLM
from utils.logging_integeration import get_logger

logger = get_logger("main", "logs/main.log")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str


app = FastAPI(
    title="PDF Chat Assistant",
    description="AI-powered PDF chat assistant",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr = OCRService()
embedding_service = EmbeddingService()
vector_store = VectorStore(dim=384)
llm = HuggingFaceLLM()

rag = RAGService(
    vector_store=vector_store,
    embedding_service=embedding_service,
    llm=llm
)

os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)


@app.get("/")
async def serve_frontend():
    return FileResponse("templates/index.html")


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process PDF file
    """
    try:
        file_path = f"data/{file.filename}"

        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"PDF saved at {file_path}")

        # OCR
        text = ocr.extract_text_from_pdf(file_path)
        logger.info("OCR completed")

        # Chunk text
        chunks = [t.strip() for t in text.split("\n") if t.strip() and len(t.strip()) > 10]

        if not chunks:
            return {"error": "No text extracted from PDF"}

        # Embeddings & indexing
        embeddings = embedding_service.get_embeddings(chunks)
        vector_store.add(embeddings, chunks)

        logger.info(f"PDF indexed successfully - {len(chunks)} chunks")

        return {
            "message": "PDF processed successfully",
            "filename": file.filename,
            "chunks_stored": len(chunks)
        }

    except Exception as e:
        logger.error(f"PDF upload error: {str(e)}")
        return {"error": f"Failed to process PDF: {str(e)}"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    """Chat with uploaded PDF"""
    question = request.question.strip()

    logger.info(f"Chat request: {question[:100]}...")

    # Safety check
    if vector_store is None or len(vector_store.text_chunks) == 0:
        return ChatResponse(
            question=question,
            answer="Please upload a PDF first before asking questions."
        )

    try:
        answer = rag.answer_question(question)
        logger.info(f"Answer generated: {answer[:100]}...")

        return ChatResponse(
            question=question,
            answer=answer
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return ChatResponse(
            question=question,
            answer=f"Sorry, I encountered an error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chunks": len(vector_store.text_chunks) if vector_store else 0
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)