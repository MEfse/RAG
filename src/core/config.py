import os
import psycopg2

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

import time

import logging
logger = logging.getLogger(__name__)

# Загружаем параметры из .env
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    # Настройки логера
    log_level: str = os.getenv("LOG_LEVEL")
    path_logging: str = os.getenv("PATH_LOGS")

    # Пути до файлов
    path_answers: str = os.getenv("PATH_ANSWERS")
    path_question: str = os.getenv("PATH_QUESTIONS")
    path_tags: str = os.getenv("PATH_TAGS")
    path_logs: str = os.getenv("PATH_LOGS")

    # Путь до промпта
    path_prompt: str = os.getenv("PATH_PROMPT")

    # Параметры подключения к базе данных
    db_name: str = os.getenv('DBNAME')
    db_user: str = os.getenv('USERNAME')
    db_password: str = os.getenv('PASSWORD')
    db_ip: str = os.getenv('HOST')
    db_port: int = os.getenv('PORT')

    # Параметры/Гиперпараметры
    batch_size: int = os.getenv('BATCH_SIZE')
    chunk_size_load: int = os.getenv('CHUNK_SIZE_LOAD')

    # Параметры Retriver
    top_chunks_retriver: int = os.getenv('TOP_CHUNKS_RETRIVER')
    min_chars: int = os.getenv('MIN_CHARS')


    # Параметры Reranker
    top_chunks_reranker: int = os.getenv('TOP_CHUNKS_RERANKER')

    DB_PARAMS: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DB_PARAMS = {
            'dbname': self.db_name,
            'user': self.db_user,  
            'password': self.db_password,
            'host': self.db_ip,
            'port': self.db_port
        }

def time_decorator(func):
    """
    Декоратор для измерения времени выполнения функции.

    Args:
        func (Callable): Функция, время выполнения которой необходимо замерить.

    Returns:
        Callable: Обертка функции, которая замеряет время выполнения.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"Ошибка при выполнении функции '{func.__name__}' за {execution_time:.4f} секунд. Ошибка: {e}", exc_info=True)
            raise
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Время выполнения функции '{func.__name__}': {execution_time:.4f} секунд")
        return result
    return wrapper
