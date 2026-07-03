import json
import logging

logger = logging.getLogger(__name__)


class RagAgent:
    def __init__(self, llm_client, search_documents_tool):
        self.llm_client = llm_client
        self.search_documents_tool = search_documents_tool

    async def run(self, query: str) -> dict:
        decision = await self._decide_tool_use(query)

        if decision["action"] == "reject":
            return {
                "answer": decision.get("reason", "Запрос отклонён."),
                "documents": [],
                "tool_used": False,
            }

        if decision["action"] == "answer_without_search":
            answer = await self.llm_client.generate(
                f"Ответь кратко на вопрос пользователя:\n{query}"
            )
            return {
                "answer": answer,
                "documents": [],
                "tool_used": False,
            }

        documents = await self.search_documents_tool.arun(query)

        context = self._build_context(documents)

        answer_prompt = f"""
            Ты RAG-ассистент. Отвечай только на основе найденных документов.

            Вопрос:
            {query}

            Документы:
            {context}

            Если в документах нет ответа, скажи:
            "В найденных документах нет достаточно информации для ответа."

            Ответ:
            """

        answer = await self.llm_client.generate(answer_prompt)

        return {
            "answer": answer,
            "documents": documents,
            "tool_used": True,
        }

    async def _decide_tool_use(self, query: str) -> dict:
        prompt = f"""
                Ты агент маршрутизации для RAG-системы.

                Твоя задача — решить, нужно ли вызывать tool search_documents.

                Доступный tool:
                - search_documents: ищет релевантные документы в базе знаний.

                Правила:
                1. Если пользователь задаёт вопрос по содержимому базы знаний, используй search_documents.
                2. Если запрос слишком короткий или бессмысленный, reject.
                3. Если это простой small-talk, можно answer_without_search.
                4. Если сомневаешься — используй search_documents.

                Верни строго JSON:
                {{
                "action": "search_documents" | "answer_without_search" | "reject",
                "reason": "краткая причина"
                }}

                Запрос пользователя:
                {query}
                """

        raw = await self.llm_client.generate(prompt)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Agent decision is not valid JSON: %s", raw)
            return {
                "action": "search_documents",
                "reason": "fallback_to_search"
            }

    def _build_context(self, documents: list[dict]) -> str:
        parts = []

        for i, doc in enumerate(documents, start=1):
            parts.append(
                f"[Документ {i}]\n"
                f"chunk_id: {doc.get('chunk_id')}\n"
                f"rerank_score: {doc.get('rerank_score')}\n"
                f"text: {doc.get('chunk_text')}"
            )

        return "\n\n".join(parts)