import pandas as pd
import numpy as np

from src.indexing.embedder import model
#from src.indexing.build_index import top_indexing



class Retriver:
    def __init__(self):
        pass

    def get_top_documents(self, data, index):
        # Преобразуем запрос в embedding
        query_embedding = model.encode("How to execute multiple queries in SQL?")

        # Поиск по индексу
        D, I = index.search(np.array([query_embedding]), k=5)

        # Получаем топ-5 похожих документов
        top_5_docs = data.iloc[I[0]]
        return top_5_docs



