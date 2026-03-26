import psycopg2

from src.ingestion.loader import DataLoader
from src.ingestion.chunker import Chunker
from src.ingestion.cleaner import Preprocessing
from src.core.config import time_decorator
from src.retrieval.retriever import Retriver
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
        self.prompt = PromptBuilder(settings)
        self.llm = LLMClient()


    @time_decorator 
    def run(self, query):
        #----------------------------------------------------------------------------------------
        logger.info("Шаг 1: Формирование запроса в базу данных.")
        #query = input(f'Введите запрос:')
        query_embedding = self.embedding_generator.encode(query)
        logger.info(f"Эмбеддинг запроса {query_embedding}.")

        #----------------------------------------------------------------------------------------
        logger.info("Шаг 2: Поиск по косинусному расстоянию ближайших чанков")
        with psycopg2.connect(**self.settings.DB_PARAMS) as conn:
            with conn.cursor() as cursor:
                chunks = self.vector.search_by_embedding(query_embedding, conn, cursor)
        logger.info(f"После preprocess: {len(chunks)} строк")
        print('Чанки', chunks)

        #----------------------------------------------------------------------------------------
        logger.info("Шаг 3: Выполенение reranker")
        chunks = self.retriver.filter_short_chunks(chunks)
        reranked_chunks = self.reranker.rerank_chunks(query, chunks)
        logger.info(f"Всего чанков: {len(reranked_chunks)}")
        print('Реранкер', reranked_chunks)

        #----------------------------------------------------------------------------------------
        logger.info("Шаг 4: Сборка контекста")
        context = self.prompt.build_context(reranked_chunks)      
        print('Контекст', context)

        #----------------------------------------------------------------------------------------
        logger.info("Шаг 5: Сборка prompt")
        prompt = self.prompt.build_prompt(query, context)   
        print('Промпт', prompt)

        #----------------------------------------------------------------------------------------
        logger.info("Шаг 6: Генерация ответа")
        answer = self.llm.generate_answer(prompt)
        print('Ответ', answer)   
        logger.info("Pipeline завершён")

        return answer