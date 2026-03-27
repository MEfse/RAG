from typing import TypedDict

from langgraph.graph import StateGraph, START, END
import pandas as pd
import psycopg2

from src.core.config import time_decorator
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.indexing.vector_store import VectorStore
from src.ingestion.chunker import Chunker
from src.ingestion.cleaner import Preprocessing
from src.ingestion.data_loader import DataLoader

import logging

from src.reranker.reranker import Reranker
from src.retrieval.retriever import Retriver
logger = logging.getLogger(__name__)

class LoaderState(TypedDict):
    questions: pd.DataFrame
    answers: pd.DataFrame
    tags: pd.DataFrame
    data: pd.DataFrame

    
class DataLoaderNode:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def __call__(self, state):
        logger.info("Node: DataLoader")

        raw_data = self.data_loader.load_data()

        return {
            "questions": raw_data["questions"],
            "answers": raw_data["answers"],
            "tags": raw_data["tags"]
        }
    

class PreprocessingNode:
    def __init__(self, preprocessing: Preprocessing):
        self.preprocessing = preprocessing

    def __call__(self, state: dict) -> dict:
        logger.info("Node: Preprocessing")

        data = self.preprocessing.preprocess_data(
            state["questions"],
            state["answers"],
            state["tags"]
        )

        logger.info("После preprocess: %s строк", len(data))

        return {"data": data}
    

class ChunkingNode:
    def __init__(self, chunker: Chunker):
        self.chunker = chunker

    def __call__(self, state: dict) -> dict:
        logger.info("Node: Chunking")

        data = state["data"]

        data["chunks"] = data["document_text"].apply(self.chunker.chunk_document)
        data["chunk_count"] = data["chunks"].apply(len)

        data = self.chunker.build_chunks_df(data)

        logger.info("Всего чанков: %s", len(data))

        return {"data": data}
    

class EmbeddingNode:
    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, state: dict) -> dict:
        logger.info("Node: Embedding")

        data = state["data"]

        embeddings = self.embedder.encode(data["chunk_text"].tolist())
        data["embedding"] = list(embeddings)

        return {"data": data}
    

def build_loader_graph(settings, embedder):

    data_loader = DataLoader(settings)
    preprocessing = Preprocessing()
    chunker = Chunker()

    builder = StateGraph(dict)

    builder.add_node("loader", DataLoaderNode(data_loader))
    builder.add_node("preprocess", PreprocessingNode(preprocessing))
    builder.add_node("chunk", ChunkingNode(chunker))
    builder.add_node("embed", EmbeddingNode(embedder))

    builder.add_edge(START, "loader")
    builder.add_edge("loader", "preprocess")
    builder.add_edge("preprocess", "chunk")
    builder.add_edge("chunk", "embed")
    builder.add_edge("embed", END)

    return builder.compile()