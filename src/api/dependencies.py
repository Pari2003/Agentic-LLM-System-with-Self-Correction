"""
FastAPI dependency injection providers.

Centralized module for dependency getters to avoid circular imports
between main.py and route modules.
"""

from __future__ import annotations

from typing import Optional

from src.models.llm_client import OllamaClient
from src.pipeline.orchestrator import Orchestrator
from src.storage.document_store import DocumentStore
from src.storage.entity_graph import EntityGraph
from src.storage.vector_store import VectorStore

# ─── Global Singletons (set by main.py at startup) ──────────────────────
_llm_client: Optional[OllamaClient] = None
_doc_store: Optional[DocumentStore] = None
_vector_store: Optional[VectorStore] = None
_graph_store: Optional[EntityGraph] = None
_orchestrator: Optional[Orchestrator] = None


def get_llm_client() -> OllamaClient:
    """FastAPI dependency: returns the shared OllamaClient instance."""
    assert _llm_client is not None, "LLM client not initialized"
    return _llm_client


def get_doc_store() -> DocumentStore:
    """FastAPI dependency: returns the shared DocumentStore instance."""
    assert _doc_store is not None, "Document store not initialized"
    return _doc_store


def get_vector_store() -> VectorStore:
    """FastAPI dependency: returns the shared VectorStore instance."""
    assert _vector_store is not None, "Vector store not initialized"
    return _vector_store


def get_graph_store() -> EntityGraph:
    """FastAPI dependency: returns the shared EntityGraph instance."""
    assert _graph_store is not None, "Entity graph not initialized"
    return _graph_store


def get_orchestrator() -> Orchestrator:
    """FastAPI dependency: returns the shared Orchestrator instance."""
    assert _orchestrator is not None, "Orchestrator not initialized"
    return _orchestrator
