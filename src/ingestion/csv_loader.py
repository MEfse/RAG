import pandas as pd
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


class CSVLoader:
    def __init__(self, default_encoding: str = "latin-1"):
        """
        Инициализация CSVLoader.

        Args:
            default_encoding (str): Кодировка по умолчанию для чтения CSV.
        """
        self.default_encoding = default_encoding


    def load_csv(
            self,
            path: str,
            encoding: str | None = None,
            drop_columns: set[str] | None = None,
            nrows: int | None = None
        ) -> pd.DataFrame:
        """
        Загружает CSV файл с возможностью исключения колонок.

        Args:
            path (str): Путь к CSV файлу.
            encoding (str | None): Кодировка файла.
            drop_columns (Iterable[str] | None): Колонки, которые нужно исключить.
            nrows (int | None): Ограничение на число строк для чтения.

        Returns:
            pd.DataFrame: Загруженные данные.

        Raises:
            FileNotFoundError: Если файл не найден.
            ValueError: Если после фильтрации не осталось колонок или чтение не удалось.
        """

        encoding = encoding or self.default_encoding
        drop_columns = set(drop_columns or ["CreationDate", "ClosedDate", "OwnerUserId"])

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        try:
            logger.info(f"Чтение заголовка файла: {path}")
            all_columns = pd.read_csv(path, encoding=encoding, nrows=1).columns.to_list()
            
            # Выбираем только нужные столбцы
            columns_to_load = [col for col in all_columns if col not in drop_columns]

            if not columns_to_load:
                raise ValueError(f"После исключения колонок в файле {path} не осталось данных для загрузки.")
            
            logger.info(
                f"Загрузка файла {path}. "
                f"Колонок для чтения: {len(columns_to_load)}. "
                f"Ограничение строк: {nrows if nrows is not None else 'без ограничения'}."
            )
            
            # Загружаем только нужные столбцы
            data = pd.read_csv(path, encoding=encoding, usecols=columns_to_load, nrows=nrows)
            
            logger.info(f'Данные из файла {path} успешно загружены, {len(data)} строк.')

            return data
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {path}: {e}", exc_info=True)
            raise ValueError(f"Не удалось загрузить CSV: {path}") from e
    
    def save_csv(self, path, data, index=False) -> None:
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

        file_path = Path(path)

        try:
            data.to_csv(file_path, index=index)
            logger.info(f'Файл сохранён в {path}.')

        except Exception as e:
            logger.error(f"Ошибка при сохранении файла {path}: {e}", exc_info=True)
            raise ValueError(f"Не удалось сохранить файл: {path}") from e

