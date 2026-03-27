from typing import Any, List, Dict

import pandas as pd
import numpy as np
from collections import defaultdict
from src.core.constants import Columns

import logging
logger = logging.getLogger(__name__)

class Retriver:
    def __init__(self, settings) -> None:
        """
        Инициализация Retriever с настройками.

        Args:
            settings: Объект настроек с минимальной длиной чанков и другими параметрами.
        """
        self.settings = settings

    def group_chunks_by_document(self, chunks):
        """
        Группирует чанки по документам (question_id, answer_id).

        Args:
            chunks (List[Dict]): Список чанков с метаданными.

        Returns:
            Dict[str, List[Dict]]: Словарь, где ключом является ID документа, а значением — список чанков.
        """

        grouped = defaultdict(list)

        for chunk in chunks:
            doc_id = f"{chunk[Columns.QUESTION_ID.value]}_{chunk[Columns.ANSWER_ID.value]}"
            grouped[doc_id].append(chunk)

        logger.info(f"Группировка чанков по документам завершена: {len(grouped)} документов.")

        return grouped
    
    def select_best_document(self, grouped_chunks) -> None | Any:
        """
        Выбирает лучший документ по минимальному расстоянию между чанками.

        Args:
            grouped_chunks (Dict[str, List[Dict]]): Группированные чанки.

        Returns:
            List[Dict]: Список чанков лучшего документа.
        """

        best_doc = None 
        best_score = float("inf")

        for doc_id, chunks in grouped_chunks.items():
            doc_score = min(c[Columns.DISTANCE.value] for c in chunks) / len(chunks)

            if doc_score < best_score:
                best_score = doc_score
                best_doc = chunks

        if best_doc is None:
            logger.warning("Не удалось выбрать лучший документ.")
        else:
            logger.info(f"Лучший документ выбран с оценкой: {best_score:.4f}")

        return best_doc
    
    def filter_short_chunks(self, chunks) -> List:
        """
        Фильтрует чанки по минимальной длине текста.

        Args:
            chunks (List[Dict]): Список чанков.

        Returns:
            List: Отфильтрованные чанки.
        """

        filtered = [chunk for chunk in chunks if len(chunk[Columns.CHUNK_TEXT.value].strip()) >= self.settings.min_chars]

        logger.info(f"После фильтра коротких чанков: {len(filtered)} из {len(chunks)}")

        return filtered

    def deduplicate_chunks(self, chunks)-> List:
        """
        Удаляет дублирующиеся чанки по тексту.

        Args:
            chunks (List[Dict]): Список чанков.

        Returns:
            List[Dict]: Список уникальных чанков.
        """

        seen = set()
        unique_chunks = []


        for chunk in chunks:
            text = chunks[Columns.CHUNK_TEXT.value].strip()

            if text not in seen:
                seen.add(text)
                unique_chunks.append(chunk)
        
        logger.info(f"После удаления дублей: {len(unique_chunks)} из {len(chunks)}")

        return unique_chunks
