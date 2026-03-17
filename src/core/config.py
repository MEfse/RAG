import os
import logging
from dotenv import load_dotenv
import psycopg2
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

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

    # Параметры подключения к базе данных
    db_name: str = os.getenv('DBNAME')
    db_user: str = os.getenv('USERNAME')
    db_password: str = os.getenv('PASSWORD')
    db_ip: str = os.getenv('HOST')
    db_port: int = os.getenv('PORT')

    # Гиперпараметры

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
