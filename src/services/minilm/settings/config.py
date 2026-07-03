"""Application settings with .env file support."""

import psycopg2
from psycopg2 import pool
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from enum import Enum

import time

import logging
logger = logging.getLogger(__name__)


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="allow")

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

    # LLM name
    MINILM: str = Field(
        ..., 
        validation_alias='MINILM'
        )

    MINILM_HOST: str = Field(
        ..., 
        validation_alias='MINILM_HOST')

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