"""
FastAPI Application — Agentic RAG QA System.

Entry point for the REST API. Manages application lifecycle (startup/shutdown),
dependency injection, and route registration.

Run with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:
    http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.agents.critic import Critic
from src.agents.generator import Generator
from src.agents.query_analyzer import QueryAnalyzer
from src.agents.refiner import Refiner
from src.agents.retriever import Retriever
from src.api import dependencies as deps
from src.config import settings
from src.models.llm_client import OllamaClient
from src.pipeline.orchestrator import Orchestrator
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.llm_reranker import LLMReranker
from src.retrieval.parent_expander import ParentExpander
from src.storage.document_store import DocumentStore
from src.storage.entity_graph import EntityGraph
from src.storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)


# ─── Application Lifecycle ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of application resources.

    Startup:
        - Initialize storage backends (SQLite, ChromaDB, Neo4j)
        - Initialize LLM client (Ollama)
        - Wire up the full agent pipeline
        - Run health check

    Shutdown:
        - Close HTTP connections
        - Close database connections
    """
    logger.info("app_startup_begin")

    # 1. Initialize storage
    deps._doc_store = DocumentStore(db_path=settings.sqlite_file)
    deps._vector_store = VectorStore(persist_dir=str(settings.chromadb_dir))

    try:
        deps._graph_store = EntityGraph(db_path=settings.sqlite_file)
        logger.info("entity_graph_connected")
    except Exception as e:
        logger.warning("entity_graph_init_failed", error=str(e))
        deps._graph_store = EntityGraph(db_path=settings.sqlite_file)

    # 2. Initialize LLM client
    deps._llm_client = OllamaClient()
    health = await deps._llm_client.health_check()
    logger.info("ollama_health", status=health.get("status"))

    # 3. Wire up the retrieval pipeline
    query_analyzer = QueryAnalyzer(deps._llm_client)
    hybrid_search = HybridSearch(deps._vector_store, deps._graph_store)
    parent_expander = ParentExpander(deps._doc_store)
    llm_reranker = LLMReranker(deps._llm_client)

    retriever = Retriever(
        llm_client=deps._llm_client,
        query_analyzer=query_analyzer,
        hybrid_search=hybrid_search,
        parent_expander=parent_expander,
        llm_reranker=llm_reranker,
    )

    # 4. Wire up generation + self-correction agents
    generator = Generator(deps._llm_client, deps._doc_store)
    critic = Critic(deps._llm_client)
    refiner = Refiner(deps._llm_client, deps._doc_store)

    # 5. Create orchestrator
    deps._orchestrator = Orchestrator(
        retriever=retriever,
        generator=generator,
        critic=critic,
        refiner=refiner,
    )

    logger.info("app_startup_complete", api_port=settings.api_port)

    yield

    # Shutdown
    logger.info("app_shutdown_begin")
    if deps._llm_client:
        await deps._llm_client.close()
    if deps._graph_store:
        deps._graph_store.close()
    logger.info("app_shutdown_complete")


# ─── App Factory ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG QA System",
    description=(
        "A production-grade agentic RAG pipeline for AI/ML Research Paper QA "
        "with semantic chunking, multi-stage retrieval, citation-backed generation, "
        "multi-layer hallucination detection, and self-correction."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─── Middleware ────────────────────────────────────────────────────────────
from src.api.middleware import setup_middleware  # noqa: E402
setup_middleware(app)


# ─── Route Registration ──────────────────────────────────────────────────
from src.api.routes import health, sessions, ingest, query, evaluation  # noqa: E402

app.include_router(health.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(evaluation.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Agentic RAG QA System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
