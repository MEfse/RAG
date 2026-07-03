from src.ingestion.data_loader import DataIngestionPipeline
from src.ingestion.postgres_ingestion_service import PostgresIngestionService
from src.core.config import time_decorator

import logging
logger = logging.getLogger(__name__)


class CSVToPostgresPipeline:
    def __init__(self, settings) -> None:
        self.settings = settings 
        self.ingestion_service = PostgresIngestionService(settings)
        self.pipeline = DataIngestionPipeline(settings)

    @time_decorator
    def run(self) -> None:
        # Преобразование CSV в PostgreSQL
        self.pipeline.run()

        