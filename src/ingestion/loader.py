import os
import pandas as pd

from src.core.config import time_decorator

import logging
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, settings):
        """
        Инициализирует DataLoader с настройками.

        Args:
            settings: Объект настроек, содержащий пути к файлам для загрузки данных.
        """
        self.settings = settings
        self.loader_csv = CSVLoader()

    @time_decorator
    def load_data(self):
        """
        Загружает данные из CSV файлов.

        Args:
            None.

        Returns:
            tuple: Тройка списков: вопросы, ответы и теги.

        Raises:
            FileNotFoundError: Если указанные файлы не найдены.
            Exception: Если произошла ошибка при загрузке данных.
        """
        try:
            # Проверка на существование файлов
            if not os.path.exists(self.settings.path_question):
                raise FileNotFoundError(f"Файл с вопросами не найден: {self.settings.path_question}")
            if not os.path.exists(self.settings.path_answers):
                raise FileNotFoundError(f"Файл с ответами не найден: {self.settings.path_answers}")
            if not os.path.exists(self.settings.path_tags):
                raise FileNotFoundError(f"Файл с тегами не найден: {self.settings.path_tags}")
            
            # Загрузка данных
            questions = self.loader_csv.load_csv(self.settings.path_question)
            answers = self.loader_csv.load_csv(self.settings.path_answers)
            tags = self.loader_csv.load_csv(self.settings.path_tags)

            return questions, answers, tags
        
        except FileNotFoundError as e:
            logger.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
            raise

class CSVLoader:
    def __init__(self, default_encoding: str = "latin-1"):
        """
        Инициализация CSVLoader.

        Args:
            default_encoding (str): Кодировка по умолчанию для чтения CSV.
        """
        self.default_encoding = default_encoding

    def load_csv(self, path, encoding=None, drop_columns=None) -> pd.DataFrame:
        """
        Загружает CSV файл с возможностью исключения колонок.

        Args:
            path (str): Путь к CSV файлу.
            encoding (str, optional): Кодировка файла.
            drop_columns (list[str], optional): Колонки, которые нужно исключить.

        Returns:
            pd.DataFrame: Загруженные данные.

        Raises:
            FileNotFoundError: Если файл не найден.
            ValueError: Если не удалось загрузить данные.
        """

        encoding = encoding or self.default_encoding
        drop_columns = drop_columns or ["CreationDate", "ClosedDate", "OwnerUserId"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")

        try:
            # Получаем все столбцы файла для фильтрации
            all_columns = pd.read_csv(path, encoding=encoding, nrows=1).columns
            
            # Выбираем только нужные столбцы
            columns_to_load = [col for col in all_columns if col not in drop_columns]
            
            # Загружаем только нужные столбцы
            data = pd.read_csv(path, encoding=encoding, usecols=columns_to_load, nrows=200)
            
            logger.info(f'Данные из файла {path} успешно загружены, {len(data)} строк.')

            return data
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {path}: {e}", exc_info=True)
            raise ValueError(f"Не удалось загрузить CSV: {path}") from e
    
    def save_csv(self, path, data, index=False):
        """
        Сохраняет DataFrame в CSV файл.

        Args:
            path (str): Путь для сохранения.
            data (pd.DataFrame): Данные для сохранения.
            index (bool): Сохранять ли индекс.

        Raises:
            TypeError: Если data не DataFrame.
            ValueError: Если произошла ошибка при сохранении.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data должен быть pandas.DataFrame")

        try:
            data.to_csv(path, index=index)
            logger.info(f'Файл сохранён в {path}.')

        except Exception as e:
            logger.error(f"Ошибка при сохранении файла {path}: {e}", exc_info=True)
            raise ValueError(f"Не удалось сохранить файл: {path}") from e

