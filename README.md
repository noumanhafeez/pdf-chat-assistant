# PDF Chat Assistant

PDF Chat Assistant is a FastAPI-powered RAG (Retrieval-Augmented Generation) application that lets users upload PDFs and chat with their content using an AI model. It uses OCR to extract text, embeddings + FAISS for semantic search, and a lightweight Hugging Face model for answering questions.

---

## Features

- Upload PDF files via a simple web interface.
- Extract text from PDFs using OCR (via `pdf2image` + `pytesseract`).
- Split extracted text into chunks and store them in a FAISS vector database.
- Generate embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`).
- Retrieve relevant text chunks for any user question.
- Generate natural‑language answers using a Hugging Face T5‑base model (`google/flan-t5-base`).
- FastAPI backend with CORS support and health‑check endpoint.
- Structured logging across all modules.

---

## Project Structure

```bash
pdf-chat-assistant/
│── app/
│   └── main.py
│
│── src/
│   ├── embeddings.py
│   ├── faiss_database.py
│   ├── hugging_face.py
│   ├── ocr_service.py
│   └── rag_pipeline.py
│
│── templates/
│   └── index.html
│
│── utils/
│   └── logging_integeration.py
│
│── data/
│── logs/
│── requirements.txt
│── README.md

```


---

## Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-name/pdf-chat-assistant.git
cd pdf-chat-assistant
```

### 2. Install system dependencies

Install `tesseract` and `pdf2image` dependencies first (platform‑specific):

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install poppler-utils
```

#### macOS (with Homebrew)

```bash
brew install tesseract
brew install poppler
```

#### Windows

Download and install:
- Tesseract OCR from: https://github.com/tesseract-ocr/tesseract
- Poppler (for `pdf2image`): https://github.com/oschwartz10612/poppler-windows

Ensure `tesseract` and `pdftoppm` are in your `PATH`.

---

### 3. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
# or
venv\Scripts\activate       # Windows
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Server

Start the FastAPI app:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- API endpoint: `http://127.0.0.1:8000`
- Web UI: `http://127.0.0.1:8000` (serves `templates/index.html`)

You can also run via the `main.py` module:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## API Endpoints

- `GET /` – Serve the frontend (`templates/index.html`).
- `POST /upload-pdf` – Upload a PDF; runs OCR, chunking, and stores embeddings.
- `POST /chat` – Ask a question about the uploaded PDF (returns `question` + `answer`).
- `GET /health` – Check health status and number of indexed chunks.

Authentication is not included; ideal for local / trusted environments.

---

## How It Works

1. **Upload PDF**  
   - PDF is saved to `data/`.
   - `OCRService` converts each page to an image and runs `pytesseract` to extract text.

2. **Text Chunking & Indexing**  
   - Text is split into lines (filtered by length).
   - `EmbeddingService` generates embeddings using `SentenceTransformer`.
   - `VectorStore` stores embeddings in FAISS and keeps the raw text chunks.

3. **Chat with RAG**  
   - `RAGService`:
     - Embeds the user question.
     - Searches FAISS for top‑k similar chunks.
     - Forms a context string from results.
   - `HuggingFaceLLM` formats a prompt and calls `google/flan-t5-base` to generate an answer.

---

## Logging

Logs are written to:

- `logs/main.log`
- `logs/embeddings.log`
- `logs/faiss_database.log`
- `logs/llm.log`
- `logs/rag_pipeline.log`
- `logs/preprocess.log`

Each module has its own logger configured via `utils/logging_integeration.py`.

---

## Example Usage

1. Go to `http://127.0.0.1:8000`.
2. Upload a PDF (e.g., a research paper or manual).
3. After upload, ask:
   - “What is this document about?”
   - “Summarize section 3.”
4. The assistant will retrieve relevant chunks and generate an answer.

---

## Requirements (Minimal)

- Python ≥ 3.8
- FastAPI + Uvicorn
- Sentence‑Transformers (`all-MiniLM-L6-v2`)
- FAISS CPU (`faiss-cpu`)
- `pdf2image` + `pytesseract`
- `transformers` pipeline (`google/flan-t5-base`)
- Tesseract OCR installed and in `PATH`

---

## Contributions

Contributions are welcome! Please open an issue or PR for:

- Improvements in OCR/text chunking.
- Better prompts or LLM integration.
- UI enhancements for `index.html`.
- Better error handling and streaming.