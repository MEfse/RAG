import json
import logging
import httpx

logger = logging.getLogger(__name__)


class MistralJudgeClient:
    def __init__(self):
        self.MODEL_NAME = "mistral:3b"
        self.OLLAMA_HOST = "http://host.docker.internal:11434/api/chat"
        self.client = httpx.AsyncClient(timeout=120)

    async def judge(self, query: str, context: str, answer: str) -> dict:
        prompt = f"""
                    Ты LLM-judge для RAG.

                    Верни строго JSON:
                    {{
                    "faithfulness": 0-1,
                    "answer_relevance": 0-1,
                    "context_relevance": 0-1,
                    "is_grounded": true/false,
                    "reason": "short"
                    }}

                    QUESTION:
                    {query}

                    CONTEXT:
                    {context}

                    ANSWER:
                    {answer}
                    """

        response = await self.client.post(
            self.OLLAMA_HOST,
            json={
                "model": self.MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "format": "json",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 256,
                    "num_ctx": 8192
                }
            }
        )

        response.raise_for_status()
        result = response.json()

        data = result.get("response")

        if isinstance(data, dict):
            return data

        try:
            return json.loads(data)
        except Exception:
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_relevance": 0.0,
                "is_grounded": False,
                "reason": "invalid json"
            }