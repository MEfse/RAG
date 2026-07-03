from collections import defaultdict


class Metrics:
    def __init__(self):
        pass

    def recall_at_k(self):
        pass

    def mrr(self):
        pass

    def f(self):
        from sklearn.metrics import ndcg_score
        import numpy as np

        # y_true — бинарный вектор релевантности (1 — релевантно, 0 — нет)
        # y_score — предсказанные скоpы (например, rrf_score или reranker score)

        y_true = np.array([[1,0,1,0]])  # пример
        y_score = np.array([[0.9,0.3,0.8,0.1]])

        ndcg = ndcg_score(y_true, y_score, k=3)

    def rrf(self, lists, k=10):
        scores = defaultdict(float)
        for lst in lists:
            for rank, doc in enumerate(lst, start=1):
                doc_id = doc["chunk_id"]
                scores[doc_id] += 1 / (k + rank)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
