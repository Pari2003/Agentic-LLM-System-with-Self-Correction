"""
Parent-Child Linker for hierarchical RAG retrieval.

Implements the "small-to-big" retrieval pattern:
1. Takes a sequence of child chunks (precise semantic splits).
2. Groups adjacent child chunks to construct larger Parent Chunks (~1024 tokens).
3. Links each child chunk to its corresponding parent chunk via parent_id.
4. Returns structured Chunk and ParentChunk objects matching our Pydantic schemas.

This structure allows the vector DB to search small, precise child embeddings,
but feeds the generator LLM the larger, context-rich parent chunk.
"""

from __future__ import annotations

import uuid

import structlog
import tiktoken

from src.config import settings
from src.models.schemas import Chunk, ChunkLevel, ChunkMetadata, ParentChunk

logger = structlog.get_logger(__name__)


class ParentChildLinker:
    """Combines child chunks into parent chunks and creates back-references."""

    def __init__(
        self,
        parent_max_tokens: int = settings.parent_chunk_max_tokens,
        encoding_name: str = "cl100k_base",
    ):
        self.parent_max_tokens = parent_max_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def _count_tokens(self, text: str) -> int:
        """Helper to count tokens in a text block."""
        return len(self.tokenizer.encode(text))

    def link(
        self,
        document_id: str,
        session_id: str,
        child_texts_with_meta: list[
            dict
        ],  # list of dicts with {"text": str, "page_number": int, "section_title": str, "chunk_type": ChunkType}
    ) -> tuple[list[Chunk], list[ParentChunk]]:
        """Group consecutive child chunks into parent chunks and establish links.

        Args:
            document_id: The ID of the document being processed.
            session_id: The current session ID (for workspace wiping).
            child_texts_with_meta: A list of dicts representing children text and metadata.

        Returns:
            A tuple of (child_chunks, parent_chunks) conforming to Pydantic schemas.
        """
        logger.info(
            "parent_child_link_start",
            document_id=document_id,
            num_children=len(child_texts_with_meta),
        )

        child_chunks: list[Chunk] = []
        parent_chunks: list[ParentChunk] = []

        current_parent_children: list[dict] = []
        current_parent_tokens = 0

        def flush_parent():
            """Helper to package the current buffer into a ParentChunk."""
            nonlocal current_parent_children, current_parent_tokens
            if not current_parent_children:
                return

            parent_id = f"parent_{uuid.uuid4()}"
            parent_text = "\n\n".join(c["text"] for c in current_parent_children)

            # Aggregate pages in the parent
            pages = sorted(list(set(c["page_number"] for c in current_parent_children)))
            # Main section title is the one from the first child in the group
            section = current_parent_children[0]["section_title"]

            # 1. Create Parent Chunk
            parent_chunk = ParentChunk(
                id=parent_id,
                text=parent_text,
                document_id=document_id,
                session_id=session_id,
                section_title=section,
                page_numbers=pages,
                child_ids=[],  # Filled in next
                token_count=current_parent_tokens,
            )

            # 2. Create Child Chunks and link to Parent ID
            for idx, child_data in enumerate(current_parent_children):
                child_id = f"child_{uuid.uuid4()}"

                metadata = ChunkMetadata(
                    document_id=document_id,
                    session_id=session_id,
                    chunk_type=child_data["chunk_type"],
                    chunk_level=ChunkLevel.CHILD,
                    section_title=child_data["section_title"],
                    page_number=child_data["page_number"],
                    page_numbers=[child_data["page_number"]],
                    parent_id=parent_id,
                    chunk_index=len(child_chunks) + 1,
                )

                child_chunk = Chunk(
                    id=child_id,
                    text=child_data["text"],
                    metadata=metadata,
                    embedding=None,  # Generated in a later phase
                    token_count=child_data["token_count"],
                )

                child_chunks.append(child_chunk)
                parent_chunk.child_ids.append(child_id)

            parent_chunks.append(parent_chunk)

            # Reset buffer
            current_parent_children = []
            current_parent_tokens = 0

        # Loop through child texts and package into parents
        for child_item in child_texts_with_meta:
            text = child_item["text"]
            tokens = self._count_tokens(text)
            child_item["token_count"] = tokens

            # If adding this child exceeds max parent tokens, flush what we have first
            if current_parent_tokens + tokens > self.parent_max_tokens and current_parent_children:
                flush_parent()

            current_parent_children.append(child_item)
            current_parent_tokens += tokens

        # Flush any remaining buffer
        flush_parent()

        logger.info(
            "parent_child_link_complete",
            document_id=document_id,
            total_parents=len(parent_chunks),
            total_children=len(child_chunks),
        )

        return child_chunks, parent_chunks
