"""
Centralized configuration management using Pydantic Settings.

All settings are configurable via environment variables or a .env file.
Sensible defaults are provided for local development with Ollama.

Usage:
    from src.config import settings
    print(settings.text_model)  # "llama3.2"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ─── Ollama Configuration ─────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    text_model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    vision_model: str = "moondream"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout: float = 120.0
    llm_max_retries: int = 3

    # ─── Semantic Chunking ────────────────────────────────────────────────
    # Child chunks: small, precise, embedded in ChromaDB for retrieval
    child_chunk_min_tokens: int = 100
    child_chunk_max_tokens: int = 512
    # Parent chunks: large, context-rich, stored in SQLite for generation
    parent_chunk_max_tokens: int = 1024
    # Overlap between consecutive child chunks (for context continuity)
    chunk_overlap_tokens: int = 50
    # Bottom N percentile of inter-sentence similarities → split boundary
    # Lower = fewer splits (larger chunks), Higher = more splits (smaller chunks)
    semantic_split_percentile: int = 25

    # ─── Retrieval Pipeline ───────────────────────────────────────────────
    # Stage 1: Hybrid search (vector + BM25 + RRF) returns this many child chunks
    hybrid_search_top_k: int = 20
    # Stage 2: After child→parent expansion and dedup, keep this many parents
    parent_expansion_top_k: int = 10
    # Stage 3: LLM reranking selects final top-k parent chunks for generation
    llm_rerank_top_k: int = 5
    # RRF fusion constant (standard value: 60)
    rrf_k: int = 60
    # Bonus RRF score for chunks containing query-related entities from the graph
    graph_boost_weight: float = 0.02
    # Relative weights for BM25 vs vector in RRF (both default to 1.0 = equal)
    bm25_weight: float = 1.0
    vector_weight: float = 1.0

    # ─── Self-Correction ──────────────────────────────────────────────────
    # Overall confidence threshold — below this triggers the Refiner agent
    confidence_threshold: float = 0.7
    # Maximum correction iterations before accepting the response as-is
    max_correction_iterations: int = 2
    # Per-claim embedding similarity threshold for hallucination flagging
    embedding_sim_threshold: float = 0.65
    # Weights for the 3 hallucination detection layers
    hallucination_embedding_weight: float = 0.4
    hallucination_entailment_weight: float = 0.4
    hallucination_keyword_weight: float = 0.2

    # ─── Session Management ───────────────────────────────────────────────
    # How long session data persists before automatic cleanup
    session_ttl_hours: int = 24
    # How often the background cleanup job runs
    cleanup_interval_minutes: int = 60

    # ─── Storage Paths ────────────────────────────────────────────────────
    chromadb_path: str = "./data/chromadb"
    sqlite_path: str = "./data/metadata.db"
    papers_dir: str = "./data/papers"

    # ─── Neo4j Configuration ──────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"

    # ─── API Server ───────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    # ─── Embedding ────────────────────────────────────────────────────────
    # nomic-embed-text produces 768-dimensional embeddings
    embedding_dim: int = 768
    # How many texts to embed in a single batch request
    embedding_batch_size: int = 32

    # ─── Pydantic Settings Config ─────────────────────────────────────────
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ─── Derived Properties ───────────────────────────────────────────────

    @property
    def chromadb_dir(self) -> Path:
        """Resolved ChromaDB storage directory."""
        path = Path(self.chromadb_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sqlite_file(self) -> Path:
        """Resolved SQLite database file path."""
        path = Path(self.sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def papers_directory(self) -> Path:
        """Resolved papers storage directory."""
        path = Path(self.papers_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def hallucination_weights(self) -> dict[str, float]:
        """Weights for hallucination detection layers as a dict."""
        return {
            "embedding": self.hallucination_embedding_weight,
            "entailment": self.hallucination_entailment_weight,
            "keyword": self.hallucination_keyword_weight,
        }


# ─── Singleton Instance ───────────────────────────────────────────────────
# Import this everywhere: `from src.config import settings`
settings = Settings()
