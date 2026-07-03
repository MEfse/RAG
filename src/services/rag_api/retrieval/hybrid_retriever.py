from settings.config import Columns
from collections import defaultdict
from typing import List

from connection.select_queries import VECTOR_QUERY, BM25_QUERY

class HybridRetriever:
    def __init__(self):
        pass

    def merge_chunks(self, vector_chunks, lexical_chunks):

        merged = {}

        for chunk in vector_chunks:
            chunk_id = chunk["chunk_id"]

            merged[chunk_id] = {
                **chunk,
                "in_vector": True,
                "in_bm25": False,
            }

        for chunk in lexical_chunks:
            chunk_id = chunk["chunk_id"]

            if chunk_id not in merged:
                merged[chunk_id] = {
                    **chunk,
                    "in_vector": False,
                    "in_bm25": True,
                }
            else:
                merged[chunk_id]["in_bm25"] = True

                if "bm25_score" in chunk:
                    merged[chunk_id]["bm25_score"] = chunk["bm25_score"]

        return list(merged.values())

    def deduplicate_chunks(self, chunks)-> List:
        seen = set()
        unique_chunks = []


        for chunk in chunks:
            chunk_id = chunk[Columns.CHUNK_ID.value].strip()

            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_chunks.append(chunk)

        return unique_chunks
    
    def rrf(self, lists, k=10):
        scores = defaultdict(float)
        for lst in lists:
            for rank, doc in enumerate(lst, start=1):
                doc_id = doc["chunk_id"]
                scores[doc_id] += 1 / (k + rank)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    def sorted_documents(self, all_chunks_list, rrf_ranked):
        # маппинг chunk_id -> полный словарь
        id_to_chunk = {doc["chunk_id"]: doc for doc in all_chunks_list}

        final_chunks = []
        for doc_id, score in rrf_ranked:            # rrf_ranked = [(chunk_id, rrf_score), ...]
            chunk = id_to_chunk[doc_id].copy()      # получаем словарь
            chunk["rrf_score"] = score              # добавляем RRF score
            final_chunks.append(chunk)

        return final_chunks


