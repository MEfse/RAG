import pandas as pd
from src.ingestion.loader import CSVLoader, DataLoader
from src.core.config import Settings
from src.pipeline import Pipeline 
import polars as pl
import psycopg2
from src.ingestion.chunker import Chunker
from src.indexing.embedder import EmbeddingGenerator
from src.ingestion.cleaner import Preprocessing
from nltk.tokenize import word_tokenize
from pathlib import Path

from src.core.queries import INSERT_CHUNK_QUERY, INSERT_CHUNK_QUERY_TEST

import logging
logger = logging.getLogger(__name__)

settings = Settings()
pipeline = Pipeline(settings)

data = pipeline.run()

with psycopg2.connect(**settings.DB_PARAMS) as conn:
    with conn.cursor() as cursor:
        rows = pipeline.db_saver.build_insert_rows(data)
        pipeline.db_saver.insert_rows(rows, conn, cursor, INSERT_CHUNK_QUERY_TEST)

