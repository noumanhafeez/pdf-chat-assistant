from transformers import pipeline
from utils.logging_integeration import get_logger

logger = get_logger("llm", "logs/llm.log")


class HuggingFaceLLM:
    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.2"
        )

    def generate_answer(self, question, context):
        logger.info("Generating answer using local LLM...")

        prompt = f"""
You are a helpful assistant. Answer using only the context.

Context:
{context}

Question:
{question}

Rules:
- Use only given context
- If not enough info, say so
- Be concise
"""

        output = self.generator(
            prompt,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.2
        )

        answer = output[0]["generated_text"]

        logger.info("Answer generated successfully")

        return answer