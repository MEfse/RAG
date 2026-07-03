import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from src.services.minilm.src.download_minilm import embedding_model


class EmbedRequest(BaseModel):
    text: str

app = FastAPI(title="MiniLM Embeddings Service")

@app.post("/embed")
def embed(request: EmbedRequest):
    vector = embedding_model.encode(request.text)
    return {"embedding": vector.tolist()}