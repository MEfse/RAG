import requests
import logging
logger = logging.getLogger(__name__)

import httpx

class QwenClient:
    def __init__(self):
        self.MODEL_NAME = "qwen2.5:3b"
        self.OLLAMA_HOST = "http://host.docker.internal:11434"
        self.client = httpx.AsyncClient(timeout=120)

    async def generate(self, prompt: str) -> str:
        response = await self.client.post(
            f"{self.OLLAMA_HOST}/api/generate",
            json={
                "model": self.MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 256,
                    "num_ctx": 4096
                }
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        logger.info(
            "Ollama stats | total=%.2fs load=%.2fs prompt_eval=%.2fs eval=%.2fs tokens=%s",
            result.get("total_duration", 0) / 1e9,
            result.get("load_duration", 0) / 1e9,
            result.get("prompt_eval_duration", 0) / 1e9,
            result.get("eval_duration", 0) / 1e9,
            result.get("eval_count")
        )

        answer = result.get("response", "").strip()

        return answer
    


class VLLMClient:
    def __init__(self):
        self.base_url = "http://host.docker.internal:8000/v1"
        self.client = httpx.AsyncClient(timeout=120)
        self.model = "Qwen/Qwen2.5-3B-Instruct"

    async def generate(self, prompt: str) -> str:
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 256
            }
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"].strip()