from src.ingestion.cleaner import DataTransformer
from src.ingestion.chunker import ChunkIndenxingService
from src.ingestion.document_builder import DocumentBuilderService

import logging

logger = logging.getLogger(__name__)

class FullCorpusPreparationPipeline:
    def __init__(self, settings, embedding_generator) -> None:
        self.settings = settings
        self.embedding_generator = embedding_generator
        
        self.preprocessing = DataTransformer()
        self.chunker = ChunkIndenxingService(settings, embedding_generator.model)
        self.document = DocumentBuilderService(settings, embedding_generator)


    def run(self) -> None:
        for documents_batch in self.chunker.get_documents(batch_size=self.settings.batch_size):
            rows = self.chunker.prepare_chunk_rows(documents_batch)
            return rows
        #self.chunker.insert_embeddings(1024, rows)
