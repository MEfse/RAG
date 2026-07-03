"""Application settings with .env file support."""

import psycopg2
from psycopg2 import pool
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from enum import Enum

import time
import inspect
from functools import wraps

import logging
logger = logging.getLogger(__name__)


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env_rag_api", 
        env_file_encoding="utf-8", 
        extra="allow")
    
    # Пути до промтов
    PATH_PROMPT_SYSTEM: str = Field(
        ..., 
        validation_alias="PATH_PROMPT_SYSTEM",
        description="Application version"
        )
    
    PATH_PROMPT_HUMAN: str = Field(
        ..., 
        validation_alias="PATH_PROMPT_HUMAN",
        description="Application version"
        )
    
    PATH_PROMPT: str = Field(
        ..., 
        validation_alias="PATH_PROMPT",
        description="Application version"
        )

    # Настройки логера
    PATH_LOGS: str = Field(
        ..., 
        validation_alias="PATH_LOGS",
        description="Application version"
        )
    
    LOG_LEVEL: str = Field(
        ..., 
        validation_alias="LOG_LEVEL",
        description="Application version"
        )

    # Individual database components
    DB_NAME: str = Field(
        ...,
        validation_alias='DBNAME',
        description="Application version"
        )
    
    DB_USER: str = Field(
        ..., 
        validation_alias='USERNAME',
        description="Application version"
        )

    DB_PASSWORD: str = Field(
        ..., 
        validation_alias='PASSWORD',
        description="Application version"
        )

    DB_IP: str = Field(
        ..., 
        validation_alias='HOST',
        description="Application version"
        )
    
    DB_PORT: int = Field(
        ..., 
        validation_alias='PORT',
        description="Application version"
        )

    # Параметры/Гиперпараметры
    BATCH_SIZE: int = Field(
        ...,
        validation_alias='BATCH_SIZE',
        description="Application version"
        )
    
    CHUNK_SIZE_LOAD: int = Field(
        ..., 
        validation_alias='CHUNK_SIZE_LOAD',
        description="Application version"
        )

    PERCENTAGE: float = Field(
        ..., 
        validation_alias='PERCENTAGE',
        description="Application version"
        )

    # Retriver settings
    TOP_CHUNKS_RETRIEVER: int = Field(
        ..., 
        validation_alias='TOP_CHUNKS_RETRIEVER'
        )
    
    MIN_CHARS: int = Field(
        ..., 
        validation_alias='MIN_CHARS')


    # Reranker settings
    TOP_CHUNKS_RERANKER: int = Field(
        ..., 
        validation_alias='TOP_CHUNKS_RERANKER')
    
    MINILM_HOST: str = Field(
        ..., 
        validation_alias='MINILM_HOST')
    
    MARCO_MINILM_HOST: str = Field(
        ..., 
        validation_alias='MARCO_MINILM_HOST')
    
    QWEN_HOST: str = Field(
        ..., 
        validation_alias='QWEN_HOST')
    
    MISTRAL_HOST: str = Field(
        ..., 
        validation_alias='MISTRAL_HOST')
    
    @property
    def DB_PARAMS(self) -> dict:
        return {
            "dbname": self.DB_NAME,
            "user": self.DB_USER,
            "password": self.DB_PASSWORD,
            "host": self.DB_IP,
            "port": self.DB_PORT,
        }

    
def time_decorator(func):
    """
    Декоратор для измерения времени выполнения функции.

    Args:
        func (Callable): Функция, время выполнения которой необходимо замерить.

    Returns:
        Callable: Обертка функции, которая замеряет время выполнения.
    """

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Ошибка при выполнении функции '{func.__name__}' "
                    f"за {execution_time:.4f} секунд. Ошибка: {e}",
                    exc_info=True
                )
                raise

            execution_time = time.time() - start_time
            logger.info(
                f"Время выполнения функции '{func.__name__}': "
                f"{execution_time:.4f} секунд"
            )
            return result

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Ошибка при выполнении функции '{func.__name__}' "
                f"за {execution_time:.4f} секунд. Ошибка: {e}",
                exc_info=True
            )
            raise

        execution_time = time.time() - start_time
        logger.info(
            f"Время выполнения функции '{func.__name__}': "
            f"{execution_time:.4f} секунд"
        )
        return result

    return sync_wrapper

class Columns(str, Enum):
    # Defaulf shared columns
    ID = "Id"
    OWNER_USER_ID = "OwnerUserId"
    CREATION_DATE = "CreationDate"
    SCORE = "Score"
    BODY = "Body"

    # Default Question columns
    CLOSED_DATE = "ClosedDate"
    TITLE = "Title"

    # Default Answers columns
    PARENT_ID = "ParentId"

    # Default Tags columns
    TAGS = "Tag"

    # Proprossed columns
    CHUNK_ID = "chunk_id"
    QUESTION_ID = "question_id"
    ANSWER_ID = "answer_id"
    BODY_QUESTION ="Body_question"
    BODY_ANSWER = "Body_answer"
    CHUNK_INDEX = "chunk_index"
    QUESTION_SCORE = "question_score"
    ANSWER_SCORE = "answer_score"
    DOCUMENT_TEXT = "document_text"
    CHUNK_TEXT = "chunk_text"
    EMBEDDING = "embedding"
    QUESTION = "Question"
    ANSWER = "Answer"