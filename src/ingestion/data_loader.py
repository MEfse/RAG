import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict
from src.core.constants import Columns
from src.ingestion.csv_loader import CSVLoader

from src.core.config import time_decorator

import logging
logger = logging.getLogger(__name__)

class DataLoader():
    def __init__(self, settings, csv_loader: CSVLoader | None = None) -> None:
        """
        Инициализирует DataLoader для загрузки CSV данных.
        
        Args:
            settings: Объект настроек, содержащий пути к файлам.
            csv_loader: Класс загрузки CSV. Если None, используется стандартный CSVLoader.
        """

        self.settings = settings
        self.csv_loader = csv_loader or CSVLoader()

    @time_decorator
    def load_data(self):
        """
        Загружает данные из CSV-файлов.
        
        Загружает три типа данных: вопросы, ответы и теги, применяя валидацию и очистку.
        
        Returns:
            RawData: Словарь с данными в формате DataFrame для каждого типа.
        
        Raises:
            FileNotFoundError: Если не найден один из файлов.
            Exception: В случае ошибок загрузки.
        """

        logger.info("Начало загрузки входных данных")

        try:
            self._validate_input_paths()

            questions = self._load_single(self.settings.path_question)
            answers = self._load_single(self.settings.path_answers)
            tags = self._load_single(self.settings.path_tags)

            questions = self._rename_columns(
                questions, 
                {
                    "Id": Columns.QUESTION_ID.value,
                    "Title":Columns.TITLE.value,
                    "Body": Columns.BODY_QUESTION.value
                }
            )

            answers = self._rename_columns(
                answers,
                    {
                        "Id": Columns.ANSWER_ID.value,
                        "ParentId": Columns.QUESTION_ID.value,
                        "Body": Columns.BODY_ANSWER.value
                    }
            )

            tags = self._rename_columns(
                tags, 
                {
                    "Id": Columns.QUESTION_ID.value,
                    "Tag": Columns.TAGS.value
                }
            )

            logger.info("Загрузка всех входных данных завершена успешно")

            return {
                "questions": questions,
                "answers": answers,
                "tags": tags
            }

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
            raise

    def _validate_input_paths(self) -> None:
        """
        Проверяет существование всех файлов данных.
        
        Raises:
            FileNotFoundError: Если какой-либо файл отсутствует.
        """

        required_paths = {
            'questions' : self.settings.path_question, 
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
        

    def _load_single(self, path: str) -> pd.DataFrame:
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
            df = self.csv_loader.load_csv(path, nrows=200)
            logger.info(f"Получено {len(df)} строк.")
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