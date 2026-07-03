import psycopg2
import logging

from src.connection.update_data_in_postgres import MERGE_QUERY
from src.settings.config import Columns
from src.ingestion.cleaner import DataTransformer
from src.ingestion.chunker import Chunker
from typing import TypedDict

logger = logging.getLogger(__name__)

class Document(TypedDict):
    title: str
    tags: str
    question: str
    answer: str

class DocumentBuilderService:
    def __init__(self, settings, embedder) -> None:
        self.settings = settings
        self.preprocessing = DataTransformer()
        self.chunker = Chunker(settings, embedder)

    def build_qa_documents(self) -> None:
        with psycopg2.connect(**self.settings.DB_PARAMS) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SET max_parallel_workers_per_gather = 0;")
                cursor.execute(MERGE_QUERY)
                conn.commit()

        logger.info("Таблица qa_documents успешно создана.")

    def build_text_for_embedding(self, doc: Document) -> str:
        """
        Формирует текст для построения эмбеддинга.

        Args:
            doc (Document): Структурированный документ

        Returns:
            str: Подготовленный текст для embedding
        """

        parts = []

        if doc[Columns.TITLE.value]:
            parts.append(f"{Columns.TITLE.value}: {doc[Columns.TITLE.value]}")

        if doc[Columns.TAGS.value]:
            parts.append(f"{Columns.TAGS.value}: {doc[Columns.TAGS.value]}")

        if doc[Columns.QUESTION.value]:
            parts.append(f"{Columns.QUESTION.value}: {doc[Columns.QUESTION.value]}")

        if doc[Columns.ANSWER.value]:
            parts.append(f"{Columns.ANSWER.value}: {doc[Columns.ANSWER.value]}")

        return "\n".join(parts).strip()
    
    def _normalize_title(self, text: str | None) -> str:
        text = text or ""
        text = self.preprocessing.delete_space(text)
        text = self.preprocessing.text_lower(text)
        return text
    
    def _normalize_body(self, text: str | None) -> str:
        text = text or ""
        text = self.preprocessing.clean_html(text)
        text = self.preprocessing.text_lower(text)
        return text
    
    def build_document_text(
        self,
        title: str | None,
        tags: str | None,
        question_body: str | None,
        answer_body: str | None
    ) -> Document:
        """
        Формирует структурированный документ из исходных полей.

        Все None значения заменяются на пустую строку.

        Args:
            title (str | None): Заголовок вопроса
            tags (str | None): Теги
            question_body (str | None): Тело вопроса
            answer_body (str | None): Лучший ответ

        Returns:
            Document: Словарь с нормализованными текстовыми полями
        """

        return {
            Columns.TITLE.value: title or "",
            Columns.TAGS.value: tags or "",
            Columns.QUESTION.value: question_body or "",
            Columns.ANSWER.value: answer_body or ""
        }
    
    def prepare_single_document(
        self,
        title: str | None,
        tags: str | None,
        question_body: str | None,
        answer_body: str | None
    ) -> list[str]:
        title = self._normalize_title(title)
        question_body = self._normalize_body(question_body)
        answer_body = self._normalize_body(answer_body)

        document = self.build_document_text(
            title=title,
            tags=tags,
            question_body=question_body,
            answer_body=answer_body
        )

        return document