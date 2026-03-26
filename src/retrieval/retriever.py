import pandas as pd
import numpy as np
from collections import defaultdict
from src.core.constants import Columns

import logging
logger = logging.getLogger(__name__)

class Retriver:
    def __init__(self, settings):
        #self.embedder = embedder
        self.settings = settings

    def group_chunks_by_document(self, chunks):
        grouped = defaultdict(list)

        for chunk in chunks:
            doc_id = f"{chunk[Columns.QUESTION_ID.value]}_{chunk[Columns.ANSWER_ID.value]}"
            grouped[doc_id].append(chunk)

        return grouped
    
    def select_best_document(self, grouped_chunks):
        best_doc = None 
        best_score = float("inf")

        for doc_id, chunks in grouped_chunks.items():
            doc_score = min(c[Columns.DISTANCE.value] for c in chunks) / len(chunks)

            if doc_score < best_score:
                best_score = doc_score
                best_doc = chunks

        return best_doc
    
    def filter_short_chunks(self, chunks):
        filtered = [chunk for chunk in chunks if len(chunk[Columns.CHUNK_TEXT.value].strip()) >= self.settings.min_chars]

        logger.info(f"После фильтра коротких чанков: {len(filtered)} из {len(chunks)}")
        return filtered

    def deduplicate_chunks(self, chunks):
        seen = set()
        unique_chunks = []


        for chunk in chunks:
            text = chunks[Columns.CHUNK_TEXT.value].strip()
            # фильтруем мусор
            if "hello and welcome to my world" in text:
                continue

            # фильтр мусора
            if "welcome to my world" in text:
                continue

            if "i'd like you to meet my family" in text:
                continue

            if text not in seen:
                seen.add(text)
                unique_chunks.append(chunk)
        
        logger.info(f"После удаления дублей: {len(unique_chunks)} из {len(chunks)}")

        return unique_chunks
