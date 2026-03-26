import pandas as pd

from src.indexing import embedder
from src.ingestion.loader import DataLoader
from src.ingestion.chunker import Chunker
from src.indexing.embedder import EmbeddingGenerator
from src.ingestion.cleaner import Preprocessing
from src.core.config import time_decorator
from src.indexing.vector_store import VectorStore
from src.core.queries import INSERT_CHUNK_QUERY, INSERT_CHUNK_QUERY_TEST

import logging
logger = logging.getLogger(__name__)

class LoaderPipeline:
    def __init__(self, settings, embedder):
        self.settings = settings
        self.data_loader = DataLoader(settings)
        self.preprocessing = Preprocessing()
        self.embedder = embedder
        self.chunker = Chunker()
        self.db_saver = VectorStore(settings)

    @time_decorator  # Применяем декоратор для замера времени
    def run(self):
        #----------------------------------------------------------------------------------------
        logger.info("Шаг 1: Загрузка данных")
        questions, answers, tags = self.data_loader.load_data()
        logger.info(f"questions={len(questions)}, answers={len(answers)}")

        #----------------------------------------------------------------------------------------
        logger.info("Шаг 2: Предобработка")
        data = self.preprocessing.preprocess_data(questions, answers, tags)
        logger.info(f"После preprocess: {len(data)} строк")

        #----------------------------------------------------------------------------------------
        logger.info("Шаг 3: Чанкинг")
        data["chunks"] = data["document_text"].apply(self.chunker.chunk_document)
        data["chunk_count"] = data["chunks"].apply(len)

        data = self.chunker.build_chunks_df(data)
        logger.info(f"Всего чанков: {len(data)}")

        #----------------------------------------------------------------------------------------
        logger.info("Шаг 4: Генерация эмбеддингов")
        embeddings = self.embedder.encode(data['chunk_text'].tolist())
        data['embedding'] = list(embeddings)
        
        logger.info("Pipeline завершён")
        return data