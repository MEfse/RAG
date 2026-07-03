class FallbackBuilder:
    def generate_rejection(self, state):
        return {
            "answer": state.get(
                "reject_reason",
                "Я могу отвечать только на вопросы по базе знаний проекта."
            )
        }
    
    async def generate_retrieval_fallback(self, state):
        return {
            "answer": (
                "Я не нашёл достаточно релевантной информации "
                "в базе знаний, чтобы уверенно ответить."
            )
        }