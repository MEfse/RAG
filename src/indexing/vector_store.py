import pandas as pd
import numpy as np
from psycopg2.extras import execute_values

from src.core.constants import Columns
from src.retrieval.retriever import Retriver
from src.core.queries import INSERT_CHUNK_QUERY, SELECT_QUERY

import logging
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, settings):
        self.settings = settings
        self.retriver = Retriver(settings)

    def build_insert_rows(self, data: pd.DataFrame):
        """
        Преобразует DataFrame с чанками в список кортежей для вставки в БД.

        Args:
            data (pd.DataFrame): DataFrame, содержащий чанки с метаданными и эмбеддингами.

        Returns:
            list[tuple]: Список кортежей, готовых для пакетной вставки в PostgreSQL.

        Raises:
            TypeError: Если data не является pandas.DataFrame.
            ValueError: Если в DataFrame отсутствуют обязательные колонки.
            ValueError: Если embedding отсутствует или имеет некорректный тип.

        Notes:
            - Использует to_records() для более быстрой итерации по данным.
            - Эмбеддинги преобразуются в list для совместимости с pgvector.
        """

        # Проверка на входные данные
        if not isinstance(data, pd.DataFrame):
            logger.error("data не является pandas.DataFrame")
            raise TypeError("Аргумент 'data' должен быть pandas.DataFrame.")
        
        logger.info(f"Получено строк: {len(data)}")
        
        # Эталонные названия колонок 
        required_columns = {
            Columns.CHUNK_ID.value,
            Columns.QUESTION_ID.value,
            Columns.ANSWER_ID.value,
            Columns.CHUNK_INDEX.value,
            Columns.TITLE.value,
            Columns.TAGS.value,
            Columns.QUESTION_SCORE.value,
            Columns.ANSWER_SCORE.value,
            Columns.CHUNK_TEXT.value,
            Columns.EMBEDDING.value,
        }

        # Проверка на пропущенные колонки
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            logger.error(f"Отсутствуют колонки: {missing_columns}")
            raise ValueError(f"В DataFrame отсутствуют обязательные колонки: {sorted(missing_columns)}")

        logger.info("Все необходимые колонки присутствуют")

        # Проверка на пустые данные
        if data.empty:
            logger.warning("DataFrame пустой — ничего не вставляем")
            return []
        
        rows = []
        
        # Проверка эмбеддингов
        for row in data.to_records(index=False): 
            embedding = row['embedding']

            if embedding is None:
                logger.error(f"embedding отсутствует для chunk_id={row[Columns.CHUNK_ID.value]}")
                raise ValueError(f"Для chunk_id={row[Columns.CHUNK_ID.value]} embedding отсутствует.")
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif isinstance(embedding, list):
                pass
            else:
                logger.error(f"Неверный тип embedding для chunk_id={row[Columns.CHUNK_ID.value]}: {type(embedding)}")
                raise ValueError(f"Для chunk_id={row[Columns.CHUNK_ID.value]} embedding должен быть np.ndarray или list, "
                                    f"получен {type(embedding).__name__}.")

        
            rows.append(
                (
                    row[Columns.CHUNK_ID.value],
                    int(row[Columns.QUESTION_ID.value]),
                    int(row[Columns.ANSWER_ID.value]),
                    int(row[Columns.CHUNK_INDEX.value]),
                    row[Columns.TITLE.value],
                    row[Columns.TAGS.value],
                    int(row[Columns.QUESTION_SCORE.value]),
                    int(row[Columns.ANSWER_SCORE.value]),
                    row[Columns.CHUNK_TEXT.value],
                    embedding
                )
            )

        logger.info(f"Сформировано {len(rows)} строк для вставки")

        if rows:
            logger.info(f"Пример строки: {rows[0]}.")

        return rows
    

    def search_by_embedding(self, query_embedding, conn, cursor): 
        """
        Выполняет поиск наиболее похожих чанков по эмбеддингу запроса.

        Args:
            query_embedding (np.ndarray | list): Вектор запроса.
            conn (psycopg2.extensions.connection): Активное соединение с БД.
            cursor (psycopg2.extensions.cursor): Курсор БД.

        Returns:
            list[dict]: Список найденных чанков.

        Raises:
            Exception: При ошибке выполнения запроса выполняется rollback.

        Notes:
            - Использует cosine similarity через pgvector (<=>).
            - Эмбеддинг преобразуется в строку перед передачей в SQL.
            - Возвращает как отдельные чанки, так и объединенный текст.
        """
        if query_embedding is None:
            logger.error("query_embedding = None")
            raise ValueError("query_embedding не должен быть None.")


        # Проверка на тип данных у эмбеддингов               
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        elif not isinstance(query_embedding, list):
            logger.error(f"Неверный тип query_embedding: {type(query_embedding)}")
            raise TypeError("query_embedding должен быть np.ndarray или list.")
        
        if len(query_embedding) == 0:
            logger.error("Пустой query_embedding")
            raise ValueError("query_embedding не должен быть пустым.")

        try:
            query_embedding_str = f"[{', '.join(map(str, query_embedding))}]"

            cursor.execute(SELECT_QUERY, (query_embedding_str, self.settings.top_chunks_retriver))

            results = cursor.fetchall()

        except Exception as e:
            # Если произошла ошибка, откатываем транзакцию и логируем ошибку
            conn.rollback()
            logger.error(f"Ошибка при получении данных из БД: {e}", exc_info=True)
            raise

        chunks = self.build_chunks_from_db_rows(results)
        logger.info(f"После parse_search_results получено чанков: {len(results)}")

        return chunks
    

    def build_chunks_from_db_rows(self, results):
        """
        Преобразует результаты SQL-запроса в список словарей с чанками.

        Args:
            results (list[tuple]): Результат cursor.fetchall(), содержащий чанки и метаданные.

        Returns:
            list[dict]: Список словарей с ключами:
                - chunk_id
                - chunk_text
                - similarity
                - question_id
                - answer_id
                - chunk_index
                - title
                - tags

        Raises:
            TypeError: Если results не является списком.

        Notes:
            Используется для последующей обработки и восстановления документа.
        """

        # Проверка на пустые данные
        if results is None:
            logger.warning("results = None")
            return []
        if len(results) == 0:
            logger.warning("Пустой результат из БД")
            return []
        
        # Проверка на тип данных
        #if isinstance(results, list):
            #logger.error(f"Неверный тип results: {type(results)}")
            #raise TypeError("Аргумент 'results' должен быть list.")

        logger.info(f"Получено результатов из БД: {len(results)}")

        chunks_data = []

        for chunk_id, chunk_text, distance, question_id, answer_id, chunk_index, title, tags in results:
            chunks_data.append(
                {
                    Columns.CHUNK_ID.value: chunk_id,
                    Columns.CHUNK_TEXT.value: chunk_text,
                    Columns.DISTANCE.value: distance,
                    Columns.QUESTION_ID.value: question_id,
                    Columns.ANSWER_ID.value: answer_id,
                    Columns.CHUNK_INDEX.value: chunk_index,
                    Columns.TITLE.value: title,
                    Columns.TAGS.value: tags,
                }
            )

        logger.info(f"Сформировано чанков: {len(chunks_data)}")

        return chunks_data
        

    def insert_rows(self, rows, conn, cursor, query=None):
        """
        Выполняет пакетную вставку чанков в базу данных.

        Args:
            rows (list[tuple]): Список кортежей для вставки.
            conn (psycopg2.extensions.connection): Активное соединение с БД.
            cursor (psycopg2.extensions.cursor): Курсор БД.

        Returns:
            None

        Raises:
            Exception: При ошибке вставки выполняется rollback и пробрасывается исключение.

        Notes:
            - Использует execute_values для высокой производительности.
            - Вставка выполняется батчами (размер задается в settings.batch_size).
            - Commit выполняется после завершения всех батчей.
        """

        query = query or INSERT_CHUNK_QUERY
        try:     
            total_rows = len(rows)

            if total_rows == 0:
                logger.warning("Нет данных для вставки в БД.")
                return 

            # Преобразуем все numpy.int64 в обычные int
            #rows = [self.convert_int64_to_int(row) for row in rows]

            for i in range(0, len(rows), self.settings.batch_size):
                batch = rows[i:i + self.settings.batch_size]

                # Выполнение пакетной вставки
                execute_values(cursor, query, batch)

                # Зафиксировать изменения
                conn.commit()

                logger.info(f"Успешно вставлено {min(i + self.settings.batch_size, total_rows)} чанков из {total_rows}.")

        except Exception as e:
            # Если произошла ошибка, откатываем транзакцию и логируем ошибку
            conn.rollback()
            logger.error(f"Ошибка при вставке чанков в базу данных: {e}", exc_info=True)
            raise           
    