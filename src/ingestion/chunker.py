from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

class Chunker:
    def __init__(self, chunk_size=500, overlap_size=100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def chunk_document(self, text):
        words = word_tokenize(text)
        chunks = []

        if not words:
            return chunks

        step = self.chunk_size - self.overlap_size
        if step <= 0:
            raise ValueError("overlap_size должен быть меньше chunk_size")

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks
    
    def build_chunks_df(self, data: pd.DataFrame) -> pd.DataFrame:
        chunks_data = []

        for _, row in data.iterrows():
            chunks = row["chunks"]

            if not chunks:
                continue

            for idx, chunk in enumerate(chunks):
                chunks_data.append(
                    {
                        "chunk_id": f"q{row['Id']}_a{row['Id_answer']}_c{idx}",
                        "question_id": row["Id"],
                        "answer_id": row["Id_answer"],
                        "chunk_index": idx,
                        "title": row["Title"],
                        "tags": row["Tag"],
                        "question_score": row["Score_question"],
                        "answer_score": row["Score_answer"],
                        "chunk_text": chunk,
                    }
                )

        return pd.DataFrame(chunks_data)