"""
ChromaDB Vector Store Adapter.

Manages child chunk embeddings and vector similarity queries.
Partitions the store by session using metadata tags, allowing:
1. Retrieval restricted to a single session (`session_id`).
2. Cleaning up all vector chunks associated with a closed session.

ChromaDB uses l2 / cosine distance metrics. We query using cosine.
"""

from __future__ import annotations

from typing import Any, Optional

import chromadb
import structlog

from src.config import settings
from src.models.schemas import Chunk, RetrievalResult

logger = structlog.get_logger(__name__)


class VectorStore:
    """Session-scoped vector store adapter wrapping ChromaDB."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or str(settings.chromadb_dir)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        # Use a single shared collection. Partitioning is handled via metadata filtering on query.
        self.collection = self.client.get_or_create_collection(
            name="child_chunks",
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity space
        )
        logger.info("chromadb_initialized", persist_dir=self.persist_dir)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add child chunks and their embeddings to the collection.

        Args:
            chunks: List of Chunk schemas with populated embeddings.
        """
        if not chunks:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            if not chunk.embedding:
                logger.error("chunk_missing_embedding", chunk_id=chunk.id)
                continue
            
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            # ChromaDB metadata must be flat: str, int, float, bool
            metadatas.append(chunk.metadata.to_chroma_metadata())

        if ids:
            logger.info("chromadb_add_chunks_start", count=len(ids))
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("chromadb_add_chunks_complete", count=len(ids))

    def query_similarity(
        self,
        query_vector: list[float],
        session_id: str,
        top_k: int = settings.hybrid_search_top_k,
    ) -> list[RetrievalResult]:
        """Perform similarity search on embeddings, filtered by session_id.

        Args:
            query_vector: The embedding of the query.
            session_id: The session to filter results.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of RetrievalResult schemas containing chunk info and distance score.
        """
        logger.debug("chromadb_query_start", session_id=session_id, top_k=top_k)

        # Force partition constraint using metadata filtering
        where_filter = {"session_id": session_id}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievalResult] = []
        
        # Check if we have results
        if not results or not results["ids"] or not results["ids"][0]:
            logger.debug("chromadb_query_empty", session_id=session_id)
            return retrieved

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            # ChromaDB returns distance. Since it's cosine space:
            # Cosine similarity = 1 - cosine_distance
            distance = distances[i]
            similarity = 1.0 - distance

            meta = metadatas[i] or {}
            
            retrieved.append(
                RetrievalResult(
                    chunk_id=ids[i],
                    parent_id=meta.get("parent_id"),
                    text=documents[i],
                    score=similarity,
                    source="vector",
                    metadata=meta,
                )
            )

        logger.debug(
            "chromadb_query_complete",
            session_id=session_id,
            results_found=len(retrieved),
        )
        return retrieved

    def delete_session_chunks(self, session_id: str) -> None:
        """Delete all vectors and metadata belonging to a session."""
        logger.info("chromadb_session_cleanup_start", session_id=session_id)
        # Delete using metadata filter
        self.collection.delete(where={"session_id": session_id})
        logger.info("chromadb_session_cleanup_complete", session_id=session_id)
