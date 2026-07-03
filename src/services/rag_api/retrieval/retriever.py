from typing import Any, List, Dict

from settings.config import Columns, time_decorator
from connection.select_queries import VECTOR_QUERY, BM25_QUERY

class Retriever:
    def __init__(self, settings, connection) -> None:
        """
        Инициализация Retriever с настройками.

        Args:
            settings: Объект настроек с минимальной длиной чанков и другими параметрами.
        """
        self.settings = settings
        self.connection = connection


    @time_decorator
    def search(self, user_query, sql_query):
        conn, cursor = self.connection._get_conn_cursor()

        if sql_query == VECTOR_QUERY:
            user_query = user_query[0]
            score_type = "vector_score"
            source_type = "vector"
        else:
            score_type = "bm25_score"
            source_type = "bm25"
    
        try:
            cursor.execute(
                sql_query, 
                (user_query, user_query, self.settings.TOP_CHUNKS_RETRIEVER)
                )
            results = cursor.fetchall()

        except Exception as e:
            conn.rollback()
            raise

        finally:
            self.connection._release_conn_cursor(conn, cursor)

        chunks = self._build_chunks_from_db_rows(
            results, 
            score_type=score_type, 
            source_type=source_type)

        return tuple(chunks)

    def _build_chunks_from_db_rows(self, results, score_type, source_type):
        if not results:
            return []

        chunks_data = []

        for chunk_id, chunk_index, chunk_text, model_name, score in results:
            chunk = {
                Columns.CHUNK_ID.value: chunk_id,
                Columns.CHUNK_INDEX.value: chunk_index,
                Columns.CHUNK_TEXT.value: chunk_text,
                "model_name": model_name,
                "source_type": source_type,
            }

            chunk[score_type] = float(score)

            chunks_data.append(chunk)

        return chunks_data