INSERT_CHUNK_QUERY = """INSERT INTO qa_chunks 
                        (chunk_id, question_id, answer_id, chunk_index, title, tags, question_score, answer_score, chunk_text, embedding)
                        VALUES %s
                        ON CONFLICT (chunk_id) DO NOTHING"""



INSERT_EMBEDDING_QUERY = """INSERT INTO {table_name}
                        (chunk_id, chunk_index, chunk_text, model_name, embedding)
                        VALUES %s
                        ON CONFLICT (chunk_id) DO NOTHING"""
