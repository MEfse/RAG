from src.core.langgragh import build_loader_graph
from src.ingestion.data_loader import DataLoader
from src.ingestion.chunker import Chunker
from src.ingestion.cleaner import Preprocessing
from src.core.config import time_decorator
from src.indexing.vector_store import VectorStore
from src.core.queries import INSERT_CHUNK_QUERY, INSERT_CHUNK_QUERY_TEST

import logging
logger = logging.getLogger(__name__)

  
class LoaderPipeline:
    def __init__(self, settings, embedder):
        self.graph = build_loader_graph(settings, embedder)

    def run(self):
        result = self.graph.invoke({})
        return result["data"]