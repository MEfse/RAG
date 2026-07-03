import traceback

from fastapi import FastAPI
from pydantic import BaseModel
from src.download_mistral import MistralJudgeClient
from settings.logging import setup_logging
setup_logging()

mistral = MistralJudgeClient()

class MistralRequest(BaseModel):
    query: str
    context: str
    answer: str

app = FastAPI(title="Mistral Service")

@app.post("/predict")
async def predict(req: MistralRequest):
    try:
        print("REQ:", req.dict())

        result = await mistral.judge(
            req.query,
            req.context,
            req.answer
        )

        print("RESULT:", result)
        return result

    except Exception as e:
        print("ERROR:", str(e))
        traceback.print_exc()
        raise