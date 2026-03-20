from sentence_transformers import SentenceTransformer

import logging
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        self.batch_data_size = 1000

    def encode(self, texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True):
        """
        Функция для получения эмбеддингов для списка текстов
        :param texts: Список строк для кодирования
        :return: Эмбеддинги для всех текстов
        """
        try:
            all_embeddings = []

            for start in range(0, len(texts), self.batch_data_size):
                batch = texts[start:start + self.batch_data_size]

                embeddings = self.model.encode(batch,
                                                batch_size=batch_size,
                                                show_progress_bar=show_progress_bar,
                                                convert_to_numpy=convert_to_numpy)

                all_embeddings.extend(embeddings)

            return all_embeddings
        
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддингов: {e}", exc_info=True)
            return None
