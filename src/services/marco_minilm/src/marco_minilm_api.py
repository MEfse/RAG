from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from src.download_marco_minilm import cross_encoder_model


class CrossEncoderRequest(BaseModel):
    pairs: List[Tuple[str, str]]

app = FastAPI(title="MarcoMiniLM CrossEncoder Service")

@app.post("/predict")
def predict(request: CrossEncoderRequest):
    scores = cross_encoder_model.predict(request.pairs)
    return {"scores": scores.tolist()}