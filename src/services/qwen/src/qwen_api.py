from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from src.download_qwen import QwenClient

from settings.logging import setup_logging

qwen = QwenClient()
setup_logging()

class QwenRequest(BaseModel):
    prompt: str

app = FastAPI(title="Qwen Service")

@app.post("/generate")
async def generate(req: QwenRequest):
    answer = await qwen.generate(req.prompt)
    return {"answer": answer}