QUESTION_QUERY = """INSERT INTO stg_questions ("Id", "OwnerUserId", "CreationDate", "ClosedDate", "Score", "Title", "Body")
                VALUES %s
                ON CONFLICT ("Id") DO NOTHING
                """

ANSWER_QUERY = """INSERT INTO stg_answers ("Id", "OwnerUserId", "CreationDate", "ParentId", "Score", "Body")
                VALUES %s
                ON CONFLICT ("Id") DO NOTHING
                """

TAG_QUERY = """INSERT INTO stg_tags ("Id", "Tag")
                VALUES %s
                ON CONFLICT ("Id") DO NOTHING
                """