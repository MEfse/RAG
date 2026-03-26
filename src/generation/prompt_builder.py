import logging
logger = logging.getLogger(__name__)

from src.core.constants import Columns


class PromptBuilder:
    def __init__(self, settings):
        self.settings = settings

    def build_context(self, chunks, add_metadata=True):
        """
        Собирает контекст для LLM из списка чанков.

        Args:
            chunks (list[dict]): Список чанков после retrieval/rerank.
            add_metadata (bool): Добавлять ли title и tags в контекст.

        Returns:
            str: Готовый текстовый контекст.
        """
        logger.info(f"Начало build_context. Получено чанков: {len(chunks)}")

        context_parts = []
        
        for idx, chunk in enumerate(chunks, start=1):
            chunk_text = chunk.get(Columns.CHUNK_TEXT.value, "").strip()
            #title = chunk.get(Columns.TITLE.value, "")
            #tags = chunk.get(Columns.TAGS.value, "")

            if not chunk_text:
                logger.warning(f"У чанка {idx} пустой chunk_text, пропускаем")
                continue

            if add_metadata:
                part = (chunk_text
                    #f"[Фрагмент {idx}]\n"
                    #f"Title: {title}\n"
                    #f"Tags: {tags}\n"
                    #f"Text: {chunk_text}"
                )
            else:
                part = f"[Фрагмент {idx}]\n{chunk_text}"

            context_parts.append(part)

        context = "\n\n".join(context_parts)

        logger.info(f"Собран context длиной {len(context)} символов")

        return context
    

    def load_prompt(self, path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
        
    def build_prompt(self, query, context):
        template = self.load_prompt(self.settings.path_prompt_test)

        prompt = template.format(query=query,context=context)

        return prompt

            
            

