from enum import Enum

class Columns(str, Enum):
    CHUNK_ID = "chunk_id"
    QUESTION_ID = "question_id"
    ANSWER_ID = "answer_id"
    BODY_QUESTION ="Body_question"
    BODY_ANSWER = "Body_answer"
    CHUNK_INDEX = "chunk_index"
    TITLE = "title"
    TAGS = "tags"
    QUESTION_SCORE = "question_score"
    ANSWER_SCORE = "answer_score"
    DOCUMENT_TEXT = "document_text"
    CHUNK_TEXT = "chunk_text"
    EMBEDDING = "embedding"
    DISTANCE = "distance"