from settings.config import Columns


import logging
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, settings) -> None:
        self.settings = settings

    def rerank_chunks(self, query, chunks, embed) -> list:
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
            logger.warning("Пустой список чанков для rerank.")
            return []
        
        pairs = [
            (
                query, 
                f"{chunk[Columns.CHUNK_INDEX.value]} {chunk[Columns.CHUNK_TEXT.value]}" 
            ) 
            for chunk in chunks
        ]

        scores = embed.get_predict(pairs)

        reranked_chunks = []
        for chunk, score in zip(chunks, scores):
            updated_chunk = chunk.copy()
            updated_chunk['rerank_score'] = float(score)
            reranked_chunks.append(updated_chunk)

        reranked_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)

        return reranked_chunks[:self.settings.TOP_CHUNKS_RERANKER]