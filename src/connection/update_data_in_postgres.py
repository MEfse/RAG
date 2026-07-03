MERGE_QUERY = '''INSERT INTO qa_documents (
                    question_id,
                    answer_id,
                    title,
                    question_body,
                    answer_body,
                    question_score,
                    answer_score,
                    tags)
                WITH tags_agg AS (
                    SELECT
                        "Id" AS question_id,
                        STRING_AGG("Tag", ', ' ORDER BY "Tag") AS tags
                    FROM stg_tags
                    GROUP BY "Id"
                )
                SELECT
                    q."Id" AS question_id,
                    a."Id" AS answer_id,
                    q."Title" AS title,
                    q."Body" AS question_body,
                    a."Body" AS answer_body,
                    q."Score" AS question_score,
                    a."Score" AS answer_score,
                    ta.tags
                FROM stg_questions q
                JOIN stg_answers a
                    ON a."ParentId" = q."Id"
                LEFT JOIN tags_agg ta
                    ON ta.question_id = q."Id";
                '''


GIN_QUERY = """CREATE INDEX idx_rag_chunks_search_vector
                ON rag_chunks
                USING GIN(search_vector);"""

TSVECTOR_QUERY = """UPDATE rag_chunks
                SET search_vector = to_tsvector('simple', coalesce(chunk_text, ''));"""

HNSW = """CREATE INDEX idx_embedding_hnsw
            ON qa_embeddings_384
            USING hnsw (embedding vector_cosine_ops);"""