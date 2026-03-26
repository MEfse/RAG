INSERT_CHUNK_QUERY = """INSERT INTO rag_chunks
                        (chunk_id, question_id, answer_id, chunk_index, title, tags, question_score, answer_score, chunk_text, embedding)
                        VALUES %s
                        ON CONFLICT (chunk_id) DO NOTHING"""

INSERT_CHUNK_QUERY_TEST = """INSERT INTO rag_chunks_test
                        (chunk_id, question_id, answer_id, chunk_index, title, tags, question_score, answer_score, chunk_text, embedding)
                        VALUES %s
                        ON CONFLICT (chunk_id) DO NOTHING"""


SELECT_QUERY = """SELECT chunk_id, chunk_text, embedding <=> %s AS similarity,
                        question_id, answer_id, chunk_index, title, tags
                    FROM rag_chunks
                    ORDER BY similarity
                    LIMIT %s;
                    """