"""
Stage 2 Parent Expansion.

Maps retrieved child chunks to their corresponding context-rich parent chunks
in SQLite. Handles deduplication and groups contributing child IDs.
"""

from __future__ import annotations

import structlog

from src.config import settings
from src.models.schemas import RankedContext, RetrievalResult
from src.storage.document_store import DocumentStore

logger = structlog.get_logger(__name__)


class ParentExpander:
    """Expands child chunk results to their parents from SQLite."""

    def __init__(self, doc_store: DocumentStore):
        self.doc_store = doc_store

    def expand(
        self,
        results: list[RetrievalResult],
        top_n: int = settings.parent_expansion_top_k,
    ) -> list[RankedContext]:
        """Map child chunks to Parent chunks, grouping and deduplicating.

        Args:
            results: List of RetrievalResult items (child chunks).
            top_n: Maximum number of parent chunks to return.

        Returns:
            List of RankedContext items containing deduplicated ParentChunks.
        """
        logger.debug("parent_expansion_start", num_children=len(results), top_n=top_n)

        parent_map: dict[str, RankedContext] = {}

        for rank_idx, child in enumerate(results):
            parent_id = child.parent_id
            if not parent_id:
                logger.warning("child_missing_parent_id", chunk_id=child.chunk_id)
                continue

            # If parent already processed, append child_id and record highest RRF score
            if parent_id in parent_map:
                context = parent_map[parent_id]
                context.contributing_child_ids.append(child.chunk_id)
                # Keep the best (highest) RRF score
                if child.score > context.rrf_score:
                    context.rrf_score = child.score
            else:
                # Fetch parent from SQLite
                parent_chunk = self.doc_store.get_parent_chunk(parent_id)
                if not parent_chunk:
                    logger.error("parent_chunk_not_found_in_store", parent_id=parent_id)
                    continue

                parent_map[parent_id] = RankedContext(
                    parent_chunk=parent_chunk,
                    rrf_score=child.score,
                    contributing_child_ids=[child.chunk_id],
                )

        # Sort by RRF score descending
        sorted_contexts = sorted(parent_map.values(), key=lambda x: x.rrf_score, reverse=True)
        top_contexts = sorted_contexts[:top_n]

        # Assign final rank indices (1-based)
        for idx, ctx in enumerate(top_contexts):
            ctx.final_rank = idx + 1

        logger.info(
            "parent_expansion_complete",
            input_children=len(results),
            output_parents=len(top_contexts),
        )
        return top_contexts
