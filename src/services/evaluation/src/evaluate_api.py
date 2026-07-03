from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from src.metrics import Metrics

from settings.logging import setup_logging
setup_logging()

metrics = Metrics()

class EvaluateRequest(BaseModel):
    text: str

app = FastAPI(title="Evaluate Service")

@app.post("/rrf")
def generate(req: EvaluateRequest):
    rff = metrics.rff(req.text)
    return {"rff": rff}