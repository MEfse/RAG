import logging
logger = logging.getLogger(__name__)

from settings.config import Columns
from langchain_core.prompts import ChatPromptTemplate


class PromptBuilder:
    def __init__(self, settings):
        self.settings = settings
        self.prompt_template = self._build_prompt_template()

    def load_prompt(self, path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()    
        
    def build_prompt(self, query, context):
        messages = self.prompt_template.format_messages(
            query=query,
            context=context
        )

        prompt_text = "\n".join([m.content for m in messages])
        return prompt_text
    
    def _build_prompt_template(self) -> ChatPromptTemplate:
        prompt_system = self.load_prompt(self.settings.PATH_PROMPT_SYSTEM)
        prompt_human = self.load_prompt(self.settings.PATH_PROMPT_HUMAN)

        return ChatPromptTemplate.from_messages([
            ("system", prompt_system),
            ("human", prompt_human)
        ])

    def build_context(self, chunks):
        logger.info(f"Начало build_context. Получено чанков: {len(chunks)}")

        context_parts = []
        
        for idx, chunk in enumerate(chunks, start=1):
            chunk_text = chunk.get(Columns.CHUNK_TEXT.value, "").strip()

            if not chunk_text:
                logger.warning(f"У чанка {idx} пустой chunk_text, пропускаем")
                continue

            else:
                part = f"[Source {idx}]\nText: {chunk_text}"

            context_parts.append(part)

        context = "\n\n".join(context_parts)

        logger.info(f"Собран context длиной {len(context)} символов")

        return context
            
            

