"""
Embeddings Generator Module.

Converts textual chunks into high-dimensional vectors (nomic-embed-text)
using the OllamaClient. Handles batching logic to prevent server timeouts.
"""

from __future__ import annotations

import structlog

from src.models.llm_client import OllamaClient
from src.models.schemas import Chunk

logger = structlog.get_logger(__name__)


class Embedder:
    """Computes vectors for text chunks using local embedding model."""

    def __init__(self, llm_client: OllamaClient):
        self.client = llm_client

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Compute embeddings for a list of Chunk objects in-place.

        Args:
            chunks: List of Chunk Pydantic schemas.

        Returns:
            The same list of chunks, with embedding fields populated.
        """
        if not chunks:
            return []

        # 1. Extract texts
        texts = [chunk.text for chunk in chunks]
        
        logger.info("embed_chunks_start", num_chunks=len(chunks))

        # 2. Get embeddings in batches
        try:
            embeddings = await self.client.embed_batch(texts)
            
            # 3. Assign back to chunk schemas
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
                
            logger.info("embed_chunks_complete", num_chunks=len(chunks))
            return chunks

        except Exception as e:
            logger.error("embed_chunks_failed", error=str(e))
            raise

