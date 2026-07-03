from typing import List, Dict, Any


class SearchDocumentsTool:
    name = "search_documents"

    def __init__(self, retrieval_pipeline):
        self.retrieval_pipeline = retrieval_pipeline

    async def arun(self, query: str) -> List[Dict[str, Any]]:
        state = await self.retrieval_pipeline.app.ainvoke(
            {"query": query}
        )

        return state.get("reranked_chunks") or state.get("chunks", [])