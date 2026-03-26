from sentence_transformers import CrossEncoder
from src.core.constants import Columns

import logging
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, settings, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Загрузка reranker-модели: {model_name}")
        self.model = CrossEncoder(model_name, device='cuda')
        self.settings = settings

    def rerank_chunks(self, query, chunks):
        """
        Переранжирует чанки по релевантности к запросу.

        Args:
            query (str): Текст запроса.
            chunks (list[dict]): Список чанков.
            top_k (int): Сколько лучших чанков вернуть.

        Returns:
            list[dict]: Пересортированные чанки.
        """
        
        if not chunks:
            logger.warning("Пустой список чанков для rerank")
            return []
        
        pairs = [(query, chunk[Columns.CHUNK_INDEX.value] + " " + chunk[Columns.CHUNK_TEXT.value]) for chunk in chunks]

        scores = self.model.predict(pairs)

        reranked_chunks = []
        for chunk, score in zip(chunks, scores):
            updated_chunk = chunk.copy()
            updated_chunk['rerank_score'] = float(score)
            reranked_chunks.append(updated_chunk)

        reranked_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        logger.info(f"После rerank оставляем top-{min(self.settings.top_chunks_reranker, len(reranked_chunks))}")

        return reranked_chunks[:self.settings.top_chunks_reranker]