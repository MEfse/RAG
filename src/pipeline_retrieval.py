from langgraph.graph import StateGraph
import psycopg2

from src.core.langgragh import END, START
from src.ingestion.data_loader import DataLoader
from src.ingestion.chunker import Chunker
from src.ingestion.cleaner import Preprocessing
from src.core.config import time_decorator
from src.retrieval.retriever import Retriver
from src.reranker.reranker import Reranker
from src.generation.llm_client import LLMClient
from src.indexing.vector_store import VectorStore
from src.generation.prompt_builder import PromptBuilder
from src.core.queries import INSERT_CHUNK_QUERY, INSERT_CHUNK_QUERY_TEST

import logging
logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self, settings, embedder, encoder):
        self.settings = settings
        self.embedding_generator = embedder
        self.encoder = encoder

        self.data_loader = DataLoader(settings)
        self.preprocessing = Preprocessing()
        self.chunker = Chunker()
        self.vector = VectorStore(settings)
        self.retriver = Retriver(settings)
        self.reranker = Reranker(settings)
        self.prompt = PromptBuilder(settings)
        self.llm = LLMClient()

        # Создаем LangGraph
        self.graph = StateGraph()
        
        # Вставляем шаги как отдельные Nodes
        self.graph.add_node("load_data", self.load_data)
        self.graph.add_node("preprocess", self.preprocess_data)
        self.graph.add_node("search_chunks", self.search_chunks)
        self.graph.add_node("rerank_chunks", self.rerank_chunks)
        self.graph.add_node("build_context", self.build_context)
        self.graph.add_node("generate_answer", self.generate_answer)

        # Связываем шаги
        self.graph.add_edge(START, "load_data")
        self.graph.add_edge("load_data", "preprocess")
        self.graph.add_edge("preprocess", "search_chunks")
        self.graph.add_edge("search_chunks", "rerank_chunks")
        self.graph.add_edge("rerank_chunks", "build_context")
        self.graph.add_edge("build_context", "generate_answer")
        self.graph.add_edge("generate_answer", END)

    @time_decorator 
    def run(self, query):
        # Запускаем граф
        result = self.graph.invoke({"query": query})
        return result["answer"]

    def load_data(self, state):
        logger.info("Шаг 1: Загрузка данных")
        raw_data = self.data_loader.load_data()
        state["raw_data"] = raw_data
        return state

    def preprocess_data(self, state):
        logger.info("Шаг 2: Предобработка")
        raw_data = state["raw_data"]
        data = self.preprocessing.preprocess_data(raw_data["questions"], raw_data["answers"], raw_data["tags"])
        state["data"] = data
        return state

    def search_chunks(self, state):
        logger.info("Шаг 3: Поиск по косинусному расстоянию ближайших чанков")
        query = state["query"]
        query_embedding = self.embedding_generator.encode(query)
        with psycopg2.connect(**self.settings.DB_PARAMS) as conn:
            with conn.cursor() as cursor:
                chunks = self.vector.search_by_embedding(query_embedding, conn, cursor)
        state["chunks"] = chunks
        return state

    def rerank_chunks(self, state):
        logger.info("Шаг 4: Выполнение reranker")
        chunks = state["chunks"]
        chunks = self.retriver.filter_short_chunks(chunks)
        reranked_chunks = self.reranker.rerank_chunks(state["query"], chunks)
        state["reranked_chunks"] = reranked_chunks
        return state

    def build_context(self, state):
        logger.info("Шаг 5: Сборка контекста")
        reranked_chunks = state["reranked_chunks"]
        context = self.prompt.build_context(reranked_chunks)
        state["context"] = context
        return state

    def generate_answer(self, state):
        logger.info("Шаг 6: Генерация ответа")
        query = state["query"]
        context = state["context"]
        prompt = self.prompt.build_prompt(query, context)
        answer = self.llm.generate_answer(prompt)
        state["answer"] = answer
        return state