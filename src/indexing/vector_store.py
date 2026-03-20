import pandas as pd
import numpy as np
from psycopg2.extras import execute_values

#from src.core.config import Settings, time_decorator
from src.core.queries import INSERT_CHUNK_QUERY, SELECT_QUERY

import logging
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, settings):
        self.settings = settings

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
            "chunk_id",
            "question_id",
            "answer_id",
            "chunk_index",
            "title",
            "tags",
            "question_score",
            "answer_score",
            "chunk_text",
            "embedding",
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
                logger.error(f"embedding отсутствует для chunk_id={row['chunk_id']}")
                raise ValueError(f"Для chunk_id={row['chunk_id']} embedding отсутствует.")
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif isinstance(embedding, list):
                pass
            else:
                logger.error(f"Неверный тип embedding для chunk_id={row['chunk_id']}: {type(embedding)}")
                raise ValueError(f"Для chunk_id={row['chunk_id']} embedding должен быть np.ndarray или list, "
                                    f"получен {type(embedding).__name__}.")

        
            rows.append(
                (
                    row["chunk_id"],
                    int(row["question_id"]),
                    int(row["answer_id"]),
                    int(row["chunk_index"]),
                    row["title"],
                    row["tags"],
                    int(row["question_score"]),
                    int(row["answer_score"]),
                    row["chunk_text"],
                    embedding,
                )
            )

        logger.info(f"Сформировано {len(rows)} строк для вставки")

        if rows:
            logger.info(f"Пример строки: {rows[0][:1]}.")

        return rows

    def parse_search_results(self, results):
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

        for chunk_id, chunk_text, similarity, question_id, answer_id, chunk_index, title, tags in results:
            chunks_data.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "similarity": similarity,
                    "question_id": question_id,
                    "answer_id": answer_id,
                    "chunk_index": chunk_index,
                    "title": title,
                    "tags": tags,
                }
            )

        logger.info(f"Сформировано чанков: {len(chunks_data)}")

        return chunks_data
    
    def restore_document(self, chunks_data):
        """
        Восстанавливает текст документа из списка чанков.

        Args:
            chunks_data (list[dict]): Список чанков с полем 'chunk_index' и 'chunk_text'.

        Returns:
            str: Восстановленный текст документа.

        Notes:
            - Чанки сортируются по chunk_index.
            - Документ собирается только из переданных чанков (не обязательно полный оригинал).
        """

        if not chunks_data:
            logger.warning("Пустой список чанков")
            return ""
        
        logger.info(f"Получено чанков: {len(chunks_data)}")

        # Сортируем чанки по chunk_index для восстановления правильного порядка
        chunks_data_sorted = sorted(chunks_data, key=lambda x: x['chunk_index'])
        logger.info("Чанки отсортированы по chunk_index")
        
        # Объединяем тексты чанков в один документ
        relevant_chunks = [chunk['chunk_text'] for chunk in chunks_data_sorted]
        full_document = " ".join(relevant_chunks)
        logger.info(f"Длина итогового документа: {len(full_document)} символов")
        
        # Включаем метаданные в восстановленный документ
        metadata = {
            'title': chunks_data_sorted[0]['title'], 
            'tags': chunks_data_sorted[0]['tags'],
            'question_id': chunks_data_sorted[0]['question_id'],
            'answer_id': chunks_data_sorted[0]['answer_id']
        }

        # Выводим метаданные и восстановленный текст
        logger.info(f"Metadata: {metadata}")
        
        return full_document
    

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
            rows = [self.convert_int64_to_int(row) for row in rows]

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

    def convert_int64_to_int(self, data):
        """
        Рекурсивно преобразует все значения типа numpy.int64 в обычные int.

        Args:
            data (any): Данные, которые могут быть numpy.int64 или обычным типом.
            
        Returns:
            any: Обработанные данные с преобразованными типами.
        """
        if isinstance(data, np.int64):
            return int(data)
        elif isinstance(data, list):
            return [self.convert_int64_to_int(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.convert_int64_to_int(item) for item in data)
        elif isinstance(data, dict):
            return {key: self.convert_int64_to_int(value) for key, value in data.items()}
        return data
            

    def select_from_db(self, query_embedding, conn, cursor): 
        """
        Выполняет поиск наиболее похожих чанков по эмбеддингу запроса.

        Args:
            query_embedding (np.ndarray | list): Вектор запроса.
            conn (psycopg2.extensions.connection): Активное соединение с БД.
            cursor (psycopg2.extensions.cursor): Курсор БД.

        Returns:
            dict: Результат поиска:
                {
                    "chunks": list[dict],   # найденные чанки
                    "document": str         # восстановленный текст из чанков
                }

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

            cursor.execute(SELECT_QUERY, (query_embedding_str,))

        except Exception as e:
            # Если произошла ошибка, откатываем транзакцию и логируем ошибку
            conn.rollback()
            logger.error(f"Ошибка при получении данных из БД: {e}", exc_info=True)
            raise

        results = cursor.fetchall()

        logger.info(f"После parse_search_results получено чанков: {len(chunks)}")

        chunks  = self.parse_search_results(results)

        return chunks 
    