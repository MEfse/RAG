import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import Json

from src.core.config import Settings

import logging
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.settings = Settings()
        self.loader_csv = CSVLoader()

    def load_data(self):
        try:
            questions = self.loader_csv.load_csv(self.settings.path_question)
            answers = self.loader_csv.load_csv(self.settings.path_answers)
            tags = self.loader_csv.load_csv(self.settings.path_tags)

            return questions, answers, tags
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
            raise

    def insert_to_db(self, data):
        try:
            # Создаем соединение с БД
            with psycopg2.connect(**self.settings.DB_PARAMS) as conn:                 
                with conn.cursor() as cursor:

                    # Запрос для вставки
                    insert_query = """INSERT INTO rag_chunks 
                        (chunk_id, question_id, answer_id, chunk_index, title, tags, question_score, answer_score, chunk_text, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

                    # Собираем данные для пакетной вставки
                    data_to_db = [(row['chunk_id'], 
                                    row['question_id'], 
                                    row['answer_id'], 
                                    row['chunk_index'], 
                                    row['title'], 
                                    row['tags'], 
                                    row['question_score'], 
                                    row['answer_score'], 
                                    row['chunk_text'], 
                                    row['embedding'].tolist()) for _, row in data.iterrows()]

                    # Выполнение пакетной вставки
                    cursor.executemany(insert_query, data_to_db)

                    # Зафиксировать изменения
                    conn.commit()

                    logger.info(f"Успешно вставлено {len(data_to_db)} чанков.")

        except Exception as e:
            logger.error(f"Ошибка при вставке чанков в базу данных: {e}", exc_info=True)
            raise
        

    def get_data_from_db(self, query_embedding):
        with psycopg2.connect(**self.settings.DB_PARAMS) as conn:
            with conn.cursor() as cursor:
                
                if isinstance(query_embedding, np.ndarray):
                    query_embedding = query_embedding.tolist()

                query_embedding_str = f"[{', '.join(map(str, query_embedding))}]"

                # Запрос для вставки
                select_query = """SELECT chunk_id, chunk_text, embedding <=> %s AS similarity
                                    FROM rag_chunks
                                    ORDER BY similarity
                                    LIMIT 10;
                                    """

                cursor.execute(select_query, (query_embedding_str,))

                results = cursor.fetchall()

                for idx, (chunk_id, chunk_text, similarity) in enumerate(results):
                    print(f"Result {idx+1}:")
                    print(f"Chunk ID: {chunk_id}")
                    print(f"Similarity: {similarity:.4f}")
                    print(f"Chunk Text: {chunk_text}")  
                    print("-" * 50)

                return results

class CSVLoader:
    def __init__(self):
        self.encoding = 'latin-1'

    def load_csv(self, path, encoding=None):
        encoding = encoding or self.encoding
        try:
            data = pd.read_csv(path, encoding=encoding)
            logger.info(f'Данные из файла {path} успешно загружены, {len(data)} строк.')
            return data
        except Exception as e:
            logger.error(f'Не удалось загрузить данные из файла {path}. Ошибка: {e}.')
            return None
    
    def save_csv(self, path, data, index=False):
        try:
            data.to_csv(path, index=index)
            logger.info(f'Файл сохранён в {path}.')
        except Exception as e:
            logger.error(f'Не удалось сохранить данные в файл {path}. Ошибка: {e}.')

