# app/services/ocr_service.py

import pytesseract
from pdf2image import convert_from_path
from utils.logging import get_logger

logger = get_logger("preprocess", "logs/preprocess.log")

class OCRService:

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        logger.info(f"Processing PDF: {pdf_path}")

        pages = convert_from_path(pdf_path)
        text = ""

        for i, page in enumerate(pages):
            page_text = pytesseract.image_to_string(page)
            text += page_text + "\n"

            logger.info(f"Processed page {i + 1}")

        return text