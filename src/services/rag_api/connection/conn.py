from contextlib import contextmanager
from typing import Any, Collection, Generator, Optional
import psycopg2 
import requests

from psycopg2 import pool
from psycopg2.extensions import connection as PGConnection, cursor as PGCursor

from settings.config import Settings


class Connection:
    def __init__(self):
        self.settings = Settings()
        self.pg_pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=1,
                        maxconn=100,
                        host=self.settings.DB_IP,
                        port=self.settings.DB_PORT,
                        dbname=self.settings.DB_NAME,
                        user=self.settings.DB_USER,
                        password=self.settings.DB_PASSWORD
                    )

    def _get_conn_cursor(self):
            conn = self.pg_pool.getconn()
            cursor = conn.cursor()
            return conn, cursor
    
    def _release_conn_cursor(self, conn, cursor):
        if cursor:
            cursor.close()

        if conn:
            self.pg_pool.putconn(conn)

class EmbeddingClient:
    def __init__(self):
        self.settings = Settings()

    def get_embedding(self, query):
        response = requests.post(
            self.settings.MINILM_HOST,
            json={"text": query}
        )
        data = response.json()
        return data.get("embedding")
    
    def get_predict(self, pairs):
        response = requests.post(
            self.settings.MARCO_MINILM_HOST,
            json={"pairs": pairs},
            timeout=30
        )

        response.raise_for_status()

        data = response.json()
        return data["scores"]
    
    def get_answer(self, prompt):
        response = requests.post(
            self.settings.QWEN_HOST,
            json={"prompt": prompt},
            timeout=30
        )

        response.raise_for_status()

        data = response.json()
        return data["answer"]
    
    async def get_score(self, query, context, answer):
        response = await requests.post(
            self.settings.MISTRAL_HOST,
            json={
                "query": query,
                "context": context,
                "answer": answer
            },
            timeout=30
        )

        response.raise_for_status()

        return response.json()
    
class PostgresClient:
    def __init__(self, db_params: dict):
        self.db_params = db_params

    @contextmanager
    def get_cursor(self) -> Generator[PGCursor, None, None]:
        conn: Optional[PGConnection] = None
        cur: Optional[PGCursor] = None
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            yield cur 
            conn.commit()
        except Exception as e:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()

    def fetch_all(self, query: str, params: tuple | None = None):
        with self.cursor as cur:
            cur.execute(query, params)
            return cur.fetchall()
        
    def fetch_one(self, query: str, params: tuple | None = None):
        with self.cursor as cur:
            cur.execute(query, params)
            return cur.fetchone()
        
    def execute(self, query: str, params: tuple | None = None):
        with self.cursor as cur:
            cur.execute(query, params)
            return cur.execute()
