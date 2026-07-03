from prometheus_client import Counter, Histogram, Gauge

REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total RAG requests",
)

REQUEST_ERRORS_TOTAL = Counter(
    "rag_request_errors_total",
    "Total RAG request errors",
)

PIPELINE_STEP_LATENCY = Histogram(
    "rag_pipeline_step_latency_seconds",
    "Latency by pipeline step",
    ["step"],
    buckets=(0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 30),
)

PIPELINE_TOTAL_LATENCY = Histogram(
    "rag_pipeline_total_latency_seconds",
    "Total RAG pipeline latency",
    buckets=(0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 30),
)

RETRIEVED_CHUNKS = Gauge(
    "rag_retrieved_chunks",
    "Number of chunks after retrieval",
)

RERANKED_CHUNKS = Gauge(
    "rag_reranked_chunks",
    "Number of chunks after reranker",
)

RETRIEVAL_BEST_RRF = Gauge(
    "rag_retrieval_best_rrf_score",
    "Best RRF score in retrieved chunks",
)

RERANK_BEST_SCORE = Gauge(
    "rag_rerank_best_score",
    "Best rerank score",
)

FALLBACKS_TOTAL = Counter(
    "rag_fallbacks_total",
    "Total fallback responses",
    ["reason"],
)

RAG_FAITHFULNESS = Gauge(
    "rag_faithfulness", 
    "LLM judge faithfulness")

RAG_RELEVANCE = Gauge(
    "rag_relevance", 
    "LLM judge relevance")

RAG_CONTEXT_USEFULNESS = Gauge(
    "rag_context_usefulness", 
    "LLM judge context usefulness")