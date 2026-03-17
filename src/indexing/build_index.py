import faiss
import numpy as np

def top_indexing(data):
    embedding_matrix = np.stack(data['embeddings'].values)

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    faiss.write_index(index, 'faiss.index')
    return index