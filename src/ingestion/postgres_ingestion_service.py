from pathlib import Path
from typing import Iterable

import pandas as pd

import psycopg2
from psycopg2.extras import execute_values

from src.ingestion.handlers import CSVChunkReader
from src.settings.config import Columns
from src.connection.from_csv_to_postgres import QUESTION_QUERY, ANSWER_QUERY, TAG_QUERY

import logging
logger = logging.getLogger(__name__)


class PostgresIngestionService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.csv_loader = CSVChunkReader()

    def load_questions_to_db(self) -> None:
        with psycopg2.connect(**self.settings.DB_PARAMS) as conn:
            with conn.cursor() as cursor:
                for chunk in self._load_single(self.settings.path_questions, chunksize=self.settings.chunk_size_load):
                    rows = [
                    (
                        int(row[Columns.ID.value]),
                        self._to_int_or_none(row[Columns.OWNER_USER_ID.value]),
                        row[Columns.CREATION_DATE.value],
                        row[Columns.CLOSED_DATE.value],
                        int(row[Columns.SCORE.value]),
                        row[Columns.TITLE.value],
                        row[Columns.BODY.value]
                    )
                    for _, row in chunk.iterrows()
                ]
                    
                    execute_values(cursor, QUESTION_QUERY, rows)

                    conn.commit()  

    def load_answer_to_db(self) -> None:
        with psycopg2.connect(**self.settings.DB_PARAMS) as conn:
            with conn.cursor() as cursor:
                for chunk in self._load_single(self.settings.path_answers, chunksize=self.settings.chunk_size_load):
                    rows = [
                    (
                        int(row[Columns.ID.value]),
                        self._to_int_or_none(row[Columns.OWNER_USER_ID.value]),
                        row[Columns.CREATION_DATE.value],
                        int(row[Columns.PARENT_ID.value]),
                        int(row[Columns.SCORE.value]),
                        row[Columns.BODY.value]
                    )
                    for _, row in chunk.iterrows()
                ]
                    
                    execute_values(cursor, ANSWER_QUERY, rows)

                    conn.commit()  

    def load_tags_to_db(self) -> None:
        with psycopg2.connect(**self.settings.DB_PARAMS) as conn:
            with conn.cursor() as cursor:
                for chunk in self._load_single(self.settings.path_tags, chunksize=self.settings.chunk_size_load):
                    rows = [
                    (
                        int(row[Columns.ID.value]),
                        row[Columns.TAGS.value]
                    )
                    for _, row in chunk.iterrows()
                ]
                    
                    execute_values(cursor, TAG_QUERY, rows)

                    conn.commit()  

    def _to_int_or_none(self, value):
        return int(value) if pd.notna(value) else None 

    def _validate_input_paths(self) -> None:
        """
        Проверяет существование всех файлов данных.
        
        Raises:
            FileNotFoundError: Если какой-либо файл отсутствует.
        """

        required_paths = {
            'questions' : self.settings.path_questions, 
            'answers' : self.settings.path_answers,
            'tags' : self.settings.path_tags
            }
        
        missing_files = [
            f'{name}: {path}' 
            for name, path in required_paths.items() 
            if not Path(path).exists()
            ]

        if missing_files:
            raise FileNotFoundError("Не найдены входные файлы:\n" + "\n".join(missing_files))
        

    def _load_single(self, path: str, chunksize: int) -> Iterable[pd.DataFrame]:
        """
        Загружает один CSV файл и возвращает его как DataFrame.

        Args:
            path (str): Путь к файлу CSV.

        Returns:
            pd.DataFrame: Данные из CSV файла.

        Raises:
            ValueError: Если файл не найден или не может быть загружен.
        """

        logger.info(f"Загрузка данных из {path}.")

        try:
            df = self.csv_loader.load_csv(path, chunksize=chunksize)
            return df
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {path}: {e}", exc_info=True)
            raise ValueError(f"Не удалось загрузить файл: {path}") from e
        
    def _rename_columns(self, df: pd.DataFrame, columns_mapping: dict) -> pd.DataFrame:
        """
        Переименовывает колонки DataFrame согласно переданному маппингу.

        Args:
            df (pd.DataFrame): Данные для переименования.
            columns_mapping (dict): Маппинг старых имен колонок на новые.

        Returns:
            pd.DataFrame: DataFrame с переименованными колонками.
        """
        return df.rename(columns=columns_mapping)