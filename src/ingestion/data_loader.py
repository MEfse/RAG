from src.ingestion.postgres_ingestion_service import PostgresIngestionService

from src.settings.config import time_decorator

import logging
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    def __init__(self, settings) -> None:
        """
        Инициализирует DataIngestionPipeline для загрузки CSV данных.
        
        Args:
            settings: Объект настроек, содержащий пути к файлам.
            csv_loader: Класс загрузки CSV. Если None, используется стандартный CSVLoader.
        """

        self.settings = settings
        self.ingestion_service = PostgresIngestionService(settings)

    def run(self) -> None:
        """
        Загружает данные из CSV-файлов.
        
        Загружает три типа данных: вопросы, ответы и теги, применяя валидацию и очистку.
        
        Returns:
            RawData: Словарь с данными в формате DataFrame для каждого типа.
        
        Raises:
            FileNotFoundError: Если не найден один из файлов.
            Exception: В случае ошибок загрузки.
        """

        logger.info("Начало загрузки входных данных.")

        try:
            # Проверка на наличие файлов
            self.ingestion_service._validate_input_paths()

            # Запись данных в БД
            self.ingestion_service.load_questions_to_db()
            self.ingestion_service.load_answer_to_db()
            self.ingestion_service.load_tags_to_db()

            logger.info("Загрузка всех входных данных завершена успешно")

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
            raise