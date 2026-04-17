from transformers import pipeline
from utils.logging_integeration import get_logger

logger = get_logger("llm", "logs/llm.log")


class HuggingFaceLLM:
    def __init__(self):
        logger.info("Loading small fast model...")

        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=-1  # CPU
        )

        logger.info("Model loaded successfully")

    def generate_answer(self, question, context):
        logger.info("Generating answer using FAST LLM...")

        prompt = f"""
You are a helpful assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer briefly and clearly.
"""

        output = self.generator(
            prompt,
            max_new_tokens=250
        )

        answer = output[0]["generated_text"]
        logger.info("Answer: {}".format(answer))

        logger.info("Answer generated successfully")

        return answer