import os
import sys
import pandas as pd
import numpy as np
import logging                          # Для логирования

# Настройка пути к проекту
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException

app = FastAPI(title="RAG Service")

# --------------------
# Health check
# --------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------
# GET /forecast (оставляем как было)
# --------------------
@app.get("/forecast")
def get_forecast():
    try:
        params = LoadParams().get_db_params()   # type: ignore
        ev = Evaluator()                        # type: ignore
        value = ev.predict(params)
        if isinstance(value, dict):
            return value
        return {"forecast": value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------
# POST /predict
# --------------------
class PredictResponse():
    forecast_start: str
    forecast_end: str
    predicted_value: float
    model_version: str

@app.post("/predict", response_model=PredictResponse)
def predict():
    pass
    try:
        params = LoadParams().get_db_params()           # type: ignore
        ev = Evaluator()                                # type: ignore
        result = ev.predict(params)

        # Если сейчас predict возвращает только число — упакуем в dict
        if not isinstance(result, dict):
            raise HTTPException(
                status_code=500,
                detail="Evaluator.predict() must return dict with forecast_start/forecast_end/predicted_value/model_version",
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------
# POST /model/reload
# --------------------
@app.post("/model/reload")
def reload_model():
    try:
        ev = Evaluator()                                            # type: ignore
        ev.reload_model()                                           # type: ignore
        return {"status": "model reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))