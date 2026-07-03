# Политики безопасности

class ModerationPolicy:
    def __init__(self, settings):
        self.settings = settings

    def moderation_guardrail(self, state):
        query = state["query"].lower()

        if len(query) < 3:
            return {
                "is_allowed": False,
                "reject_reason": "Слишком короткий запрос."
            }

        banned_patterns = [
            "ignore previous instructions",
            "system prompt",
            "delete database",
            "drop table",
        ]

        if any(pattern in query.lower() for pattern in banned_patterns):
            return {
                "is_allowed": False,
                "reject_reason": "Запрос нарушает правила безопасности."
            }

        return {
            "is_allowed": True
        }
    
    def retrieval_quality_check(self, state):
        chunks = state.get("chunks", [])

        if not chunks:
            return {
                "retrieval_passed": False,
                "reject_reason": "Не нашёл релевантных документов."
            }

        top_score = chunks[0].get("rrf_score", 0)

        if top_score < 0.02:
            return {
                "retrieval_passed": False,
                "reject_reason": "Найденные документы слишком слабо связаны с запросом."
            }

        return {
            "retrieval_passed": True
        }
    
    async def check_retrieval_quality(self, state):
        chunks = state.get("chunks", [])

        if not chunks:
            return {
                "is_allowed": False,
                "fallback_reason": "no_chunks"
            }

        best_rrf = max(c.get("rrf_score", 0.0) for c in chunks)
        hybrid_hits = sum(
            1 for c in chunks
            if c.get("in_vector") and c.get("in_bm25")
        )

        # стартовые пороги, потом подберёшь по логам
        if best_rrf < 0.02 and hybrid_hits == 0:
            return {
                "is_allowed": False,
                "fallback_reason": "low_retrieval_confidence"
            }

        return {
            "is_allowed": True,
            "fallback_reason": ""
        }
    