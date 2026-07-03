# Библиотека для создания графа
from langgraph.graph import StateGraph
from langgraph.func import START, END

# Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.retriever import Retriever

# Reranker
from reranker.reranker import Reranker

# Промпты и ответ LLM
from prompt.prompt_builder import PromptBuilder
from connection.conn import EmbeddingClient
from connection.insert_queries import INSERT_CHUNK_QUERY
from connection.select_queries import VECTOR_QUERY, BM25_QUERY

from monitoring.metrics import (
    REQUESTS_TOTAL,
    REQUEST_ERRORS_TOTAL,
    PIPELINE_TOTAL_LATENCY, 
    PIPELINE_STEP_LATENCY,
    RAG_FAITHFULNESS,
    RAG_RELEVANCE,
    RAG_CONTEXT_USEFULNESS
)

# Для посчета времени выполения функций и запросы в БД.
from settings.config import time_decorator
from typing import TypedDict, Any

# Логгирование и настройки
import logging
import time
from settings.config import Settings
logger = logging.getLogger(__name__)

import asyncio


class RetrievalState(TypedDict, total=False):
    query: str
    raw_data: Any
    data: Any
    chunks: list[dict]
    reranked_chunks: list[dict]
    context: str
    answer: str
    is_allowed: bool
    reject_reason: str
    retrieval_stats: dict
    fallback_reason: str


class RetrievalPipeline:
    def __init__(
        self,
        settings: Settings,
        embed: EmbeddingClient,
        retriever: Retriever,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        prompt: PromptBuilder,
        moderation_policy,
        fallback_builder) -> None:

        self.settings = settings
        self.embed = embed
        self.retriever = retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.prompt = prompt
        self.moderation_policy = moderation_policy
        self.fallback_builder = fallback_builder

        # Создаем LangGraph
        self.graph = StateGraph(RetrievalState)
        
        # Вставляем шаги как отдельные Nodes
        self.graph.add_node("moderation_guardrail", self.moderation_policy.moderation_guardrail)
        self.graph.add_node("generate_rejection", self.fallback_builder.generate_rejection)

        self.graph.add_node("retriever_only", self.retriever_only)

        self.graph.add_node("check_retrieval_quality", self.moderation_policy.check_retrieval_quality)
        self.graph.add_node("generate_retrieval_fallback", self.fallback_builder.generate_retrieval_fallback)

        self.graph.add_node("reranker_only", self.reranker_only)
        self.graph.add_node("build_context", self.build_context)
        self.graph.add_node("generate_answer", self.generate_answer)

        # Связываем шаги
        self.graph.add_edge(START, "moderation_guardrail")

        self.graph.add_conditional_edges(
            "moderation_guardrail",
            self.route_after_guardrail,
            {
                "retriever_only": "retriever_only",
                "generate_rejection": "generate_rejection"
            }
        )

        self.graph.add_edge("generate_rejection", END)

        self.graph.add_edge("retriever_only", "check_retrieval_quality")

        self.graph.add_conditional_edges(
            "check_retrieval_quality",
            self.route_after_retrieval_check,
            {
                "reranker_only": "reranker_only",
                "generate_retrieval_fallback": "generate_retrieval_fallback"
            }
        )

        self.graph.add_edge("generate_retrieval_fallback", END)
        self.graph.add_edge("reranker_only", "build_context")
        self.graph.add_edge("build_context", "generate_answer")
        self.graph.add_edge("generate_answer", END)


        self.app = self.graph.compile()

    async def run(self, query):
        REQUESTS_TOTAL.inc()
        start = time.time()

        try:
            state = await self.app.ainvoke({"query": query})
            return {
                "query": state["query"],
                "answer": state.get("answer")
            }
        except Exception:
            REQUEST_ERRORS_TOTAL.inc()
            raise
        finally:
            PIPELINE_TOTAL_LATENCY.observe(time.time() - start)

    async def retriever_only(self, state: RetrievalState):
        start = time.time()

        try:
            logger.info("Шаг 1: Retriever")
            query = state["query"]

            query_embedding = self.embed.get_embedding(query)

            vector_task = asyncio.to_thread(
                self.retriever.search,
                query_embedding, 
                VECTOR_QUERY
            )

            lexical_task = asyncio.to_thread(
                self.retriever.search,
                query, 
                BM25_QUERY
            )

            vector_chunks, lexical_chunks = await asyncio.gather(
                vector_task,
                lexical_task
            )

            chunks = self.hybrid_retriever.merge_chunks(
                vector_chunks, lexical_chunks
                )

            rrf = self.hybrid_retriever.rrf(
                [vector_chunks, lexical_chunks]
                )

            chunks = self.hybrid_retriever.sorted_documents(
                chunks, rrf
                )
            
            return {"chunks": chunks}
        
        finally:
            PIPELINE_STEP_LATENCY.labels("retriever").observe(
                time.time() - start
            )
    
    def reranker_only(self, state: RetrievalState):
        start = time.time()

        try:
            logger.info("Шаг 2: Выполнение reranker")

            reranked_chunks = self.reranker.rerank_chunks(
                state["query"], 
                state["chunks"], 
                self.embed
                )

            return {"reranked_chunks": reranked_chunks}
        
        finally:
            PIPELINE_STEP_LATENCY.labels("reranker").observe(
                time.time() - start
            )

    def build_context(self, state: RetrievalState):
        start = time.time()

        try:
            logger.info("Шаг 3: Сборка контекста")

            context = self.prompt.build_context(state["reranked_chunks"])

            return {"context": context}
        
        finally:
            PIPELINE_STEP_LATENCY.labels("context").observe(
                time.time() - start
            )

    async def generate_answer(self, state):
        start = time.time()

        try:
            logger.info("Шаг 4: Генерация ответа")

            prompt = self.prompt.build_prompt(
                state["query"],
                state["context"]
            )

            answer = self.embed.get_answer(prompt)

            # judge_result = await self.embed.get_score(
            #     query=state["query"],
            #     context=state.get("context", ""),
            #     answer=answer
            # )

            # RAG_FAITHFULNESS.set(judge_result["faithfulness"])
            # RAG_RELEVANCE.set(judge_result["relevance"])
            # RAG_CONTEXT_USEFULNESS.set(judge_result["context_usefulness"])

            return {
                "prompt": prompt,
                "answer": answer
            }
        
        finally:
            PIPELINE_STEP_LATENCY.labels("llm").observe(
                time.time() - start
            )
    

    def route_after_guardrail(self, state):
        if state.get("is_allowed"):
            return "retriever_only"

        return "generate_rejection"
    
    def route_after_retrieval_check(self, state):
        if state.get("retrieval_passed"):
            return "reranker_only"

        return "generate_rejection"
    
    def route_after_retrieval_check(self, state):
        if state.get("is_allowed"):
            return "reranker_only"

        return "generate_retrieval_fallback"