from fastapi import FastAPI, Response
from pydantic import BaseModel

# Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.retriever import Retriever

# Reranker
from reranker.reranker import Reranker

# Промпты и ответ LLM
from prompt.prompt_builder import PromptBuilder
from connection.conn import Connection, EmbeddingClient

from pipeline.polities import ModerationPolicy
from pipeline.fallback import FallbackBuilder

from settings.config import Settings
from settings.logging import setup_logging
from pipeline.pipeline_retrieval import RetrievalPipeline

from prometheus_fastapi_instrumentator import Instrumentator # type: ignore
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

settings = Settings()
setup_logging(settings)

class QueryRequest(BaseModel):
    query: str

app = FastAPI(title='RAG API Service')
Instrumentator().instrument(app).expose(app)

retrieval = RetrievalPipeline(
    settings=settings,
    embed=EmbeddingClient(),
    retriever=Retriever(settings, Connection()),
    hybrid_retriever=HybridRetriever(),
    reranker=Reranker(settings),
    prompt=PromptBuilder(settings),
    moderation_policy=ModerationPolicy(settings),
    fallback_builder=FallbackBuilder()
    )

@app.post('/test')
async def query(req: QueryRequest):
    return await retrieval.run(req.query)

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )