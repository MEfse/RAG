VECTOR_QUERY = """SELECT
                    chunk_id,
                    chunk_index,
                    chunk_text,
                    model_name,
                    embedding <=> %s::vector AS distance
                FROM qa_embeddings_384
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """

BM25_QUERY = """SELECT chunk_id, chunk_index, chunk_text, model_name, ts_rank(search_vector, plainto_tsquery('simple', %s)) AS lexical_score
                FROM qa_embeddings_384
                WHERE search_vector @@ plainto_tsquery('simple', %s)
                ORDER BY lexical_score DESC
                LIMIT %s;"""

QA_QUERY = """WITH BestAnswers AS (
                SELECT
                    question_id,
                    MAX(answer_score) AS max_score
                FROM qa_documents
                GROUP BY question_id
            )
                SELECT
                    q.question_id,
                    q.answer_id,
                    q.title,
                    q.question_body,
                    q.answer_body,
                    q.question_score,
                    q.answer_score,
                    q.tags
                FROM qa_documents q
                JOIN BestAnswers ba
                    ON q.question_id = ba.question_id
                    AND q.answer_score = ba.max_score;"""