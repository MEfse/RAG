from dataclasses import dataclass
from typing import List, Tuple
from collections.abc import Generator

import psycopg2
from psycopg2.extras import execute_values
from src.ingestion.cleaner import DataTransformer

from src.connection.select_queries import QA_QUERY
from src.connection.insert_queries import INSERT_EMBEDDING_QUERY

import math

import logging
logger = logging.getLogger(__name__)
   
@dataclass
class EmbeddingRow:
    chunk_id: str
    chunk_index: int
    chunk_text: str
    model_name: str
    embedding: list[float]

class Chunker:
    def __init__(self, settings, embedder) -> None:
        self.embedder = embedder
        self.settings = settings
        self.percentage = settings.percentage

        self.tokenizer = self.embedder.tokenizer

    def chunk_text(self, document: list) -> list[str]:
        """
        Разбивает текст на чанки по токенам tokenizer'а модели
        с учетом overlap между соседними чанками.

        Args:
            text (str): Исходный текст для чанкования.

        Returns:
            list[str]: Список чанков в текстовом виде.

        Raises:
            ValueError: Если overlap_size больше или равен chunk_size.
        """
        title = document.get('Title', '')
        tag = document.get('Tag', '')
        question_text = document.get('Question', '')
        answer_text = document.get('Answer', '')

        full_text = f"Title: {title} Tag: {tag} Question: {question_text} Answer: {answer_text}"

        chunk_size = self.embedder.default_embedding_dim
        overlap_size = math.floor(chunk_size * self.percentage)

        if not full_text:
            return []

        step = chunk_size - overlap_size
        if step <= 0:
            raise ValueError("overlap_size должен быть меньше chunk_size.")
        
        tokens = self.tokenizer.tokenize(full_text)
        if not tokens:
            return []

        chunks = []

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            if chunk:
                chunks.append(chunk)

        return chunks

class ChunkIndenxingService:  
    def __init__(self, settings, embedder) -> None:
        from src.ingestion.document_builder import DocumentBuilderService
        self.settings = settings
        self.preprocessing = DataTransformer()
        self.chunker = Chunker(settings, embedder)
        self.embedder = embedder
        self.document_builder = DocumentBuilderService(settings, embedder)

        self.model_name = self.embedder.model_name 
        self.tokenizer = self.embedder.tokenizer

    def get_documents(self, batch_size: int) -> Generator[list[tuple], None, None]:
        """
        Извлекает документы из базы данных по пакетам.

        Эта функция использует курсор с итерацией по пакету (`fetchmany`) для извлечения данных
        из базы данных. Она возвращает генератор, который поочередно передает пакеты данных.

        Args:
            batch_size (int): Размер пакета для извлечения данных.

        Returns:
            Generator[list[tuple], None, None]: Генератор, который возвращает пакеты данных (списки строк).
        
        Exceptions:
            ValueError: Если batch_size меньше или равно нулю.
            psycopg2.Error: Если возникает ошибка при взаимодействии с базой данных.
            Exception: Общие ошибки в процессе выполнения.
        """

        if batch_size <= 0:
            logger.error(f"Неверный размер пакета: {batch_size}. Размер пакета должен быть больше нуля.")
            raise ValueError("batch_size должен быть больше нуля.")
    
        try:
            # Создание соединения с базой данных
            with psycopg2.connect(**self.settings.DB_PARAMS) as conn:
                with conn.cursor(name="qa_documents_cursor") as cursor:
                    cursor.itersize = batch_size
                    cursor.execute(QA_QUERY)

                    # Логируем начало извлечения данных
                    logger.info(f"Извлечение документов с размером пакета {batch_size}...")

                    # Извлечение данных по пакету
                    while True:
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break

                        # Логируем количество строк в пакете
                        logger.info(f"Получено {len(rows)} строк.")
                        yield rows

        except psycopg2.Error as e:
            # Логирование ошибок при взаимодействии с PostgreSQL
            logger.error(f"Ошибка при получении данных из базы: {e.pgcode} - {e.pgerror}", exc_info=True)
            raise
        except Exception as e:
            # Логирование общих ошибок
            logger.error(f"Ошибка получения документа: {e}", exc_info=True)
            raise 


    def prepare_chunk_rows(self, documents_batch: List[Tuple]) -> list[EmbeddingRow]:
        """
        Подготавливает строки для вставки эмбеддингов в БД.

        Args:
            documents_batch (List[Tuple]): Пакет документов из БД.

        Returns:
            List[Tuple]: Строки для массовой вставки эмбеддингов.
        """
        rows_to_insert: List[Tuple] = []

        for row in documents_batch:
            (
                question_id,
                answer_id,
                title,
                question_body,
                answer_body,
                question_score,
                answer_score,
                tags,
            ) = row

            # Формирование документа
            document = self.document_builder.prepare_single_document(title, tags, question_body, answer_body) 
            # Режем документ
            chunk_document = self.chunker.chunk_text(document)

            embeddings = self.embedder.encode(chunk_document, show_progress_bar=False)

            for chunk_index, (chunk_text, embedding) in enumerate(zip(chunk_document, embeddings)):
                chunk_id = f"q{question_id}_a{answer_id}_c{chunk_index}"
                
                rows_to_insert.append(
                    EmbeddingRow(
                        str(chunk_id),
                        int(chunk_index),
                        str(chunk_text),
                        str(self.embedder.model_name),
                        [float(x) for x in embedding]
                    )
                )

        return rows_to_insert       
    
    def insert_embeddings(self, table_suffix: int,  data: list[tuple]) -> None:
        """
        Массово вставляет эмбеддинги в таблицу qa_embeddings_<table_suffix>.

        Args:
            table_suffix (str): Суффикс таблицы, обычно размерность эмбеддинга
            data (list[tuple]): Список строк для вставки.

        Raises:
            ValueError: Если table_suffix не поддерживается.
            psycopg2.Error: Если произошла ошибка при вставке в БД.
        """

        table_suffix_str = self._embeddings_len(table_suffix)

        table_name = f"qa_embeddings_{table_suffix_str}"

        query = INSERT_EMBEDDING_QUERY.format(table_name=table_name)

        try:
            with psycopg2.connect(**self.settings.DB_PARAMS) as conn:
                with conn.cursor() as cursor:
                    rows_as_tuples = [
                        (row.chunk_id, row.chunk_index, row.chunk_text, row.model_name, row.embedding)
                        for row in data
                        ]
                    execute_values(cursor, query, rows_as_tuples)
                    conn.commit()

            logger.info(f"Данные сохранены в таблице qa_embeddings_{table_suffix}.")

        except psycopg2.Error as e:
            logger.error(
                f"Ошибка вставки в qa_embeddings_{table_suffix}: {e}",
                exc_info=True
            )
            raise

    def _embeddings_len(self, table_suffix) -> str:
        if table_suffix <= 2 ** 8: 
            return str(2 ** 8)          # 256
        
        if table_suffix > 2 ** 8 and table_suffix <= 384: 
            return str(384)             # 384
        
        if table_suffix > 384 and table_suffix <= 2 ** 9: 
            return str(2 ** 9)          # 512
        
        if table_suffix > 2 ** 9 and table_suffix <= 2 ** 10: 
            return str(2 ** 10)         # 1024
        
        if table_suffix > 2 ** 10 and table_suffix <= 2 ** 11: 
            return str(2 ** 11)         # 2048
        
        if table_suffix > 2 ** 11 and table_suffix <= 2 ** 12: 
            return str(2 ** 12)         # 4096


            