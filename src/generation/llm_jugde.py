import ollama
from ollama import Client
import logging
import re

from settings.config import time_decorator

logger = logging.getLogger(__name__)

class LLMJudge:
    def __init__(self, model_name="qwen2.5:3b"): # Убедитесь, что модель скачана: ollama pull mistral
        self.model_name = model_name
        # Используем Client для взаимодействия с API
        self.client = Client(host='http://localhost:11434')

    #@time_decorator
    def generate_answer(self, query: str, document: str) -> float:
        # Промпт с четким указанием формата
        system_prompt = (
            "You are an expert LLM judge. Your task is to compare a query with a document "
            "and return a relevance score between 0 and 1. 0 = completely irrelevant, "
            "1 = fully relevant. Respond only with a number, no explanations."
        )
        user_prompt = f"Query: {query}\nDocument: {document}\nRelevance score (0 to 1):"

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                options={'temperature': 0.2} # Ставим 0 для стабильности оценки
            )
            
            content = response['message']['content'].strip()
            logger.debug(f"LLM raw output: {content[:100]}")  # первые 100 символов
            match = re.search(r"(\d+(\.\d+)?)", content)
            
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.0
            
        except Exception as e:
            logger.error(f"Error during LLM judging: {e}")
            return 0.0

    @time_decorator
    def evaluate_chunks(self, query: str, chunks: list) -> list:
        """
        Оценивает список chunks и добавляет ключ 'llm_score'
        """
        results = []
        for chunk in chunks:
            # Важно: используем копию, чтобы не менять исходные данные вне функции
            updated_chunk = chunk.copy()
            doc_text = updated_chunk.get("text", "")
            score = self.generate_answer(query, doc_text)
            updated_chunk["llm_score"] = score
            results.append(updated_chunk)
        return results
