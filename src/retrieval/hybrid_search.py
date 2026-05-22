"""
Stage 1 Hybrid Retrieval.

Combines Vector Search (dense embeddings) and BM25 (sparse keyword search)
using Reciprocal Rank Fusion (RRF), with a score boost for chunks that
contain query-related entities/relationships from the Neo4j Knowledge Graph.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import structlog
from rank_bm25 import BM25Okapi

from src.config import settings
from src.models.schemas import RetrievalResult
from src.storage.vector_store import VectorStore
from src.storage.entity_graph import EntityGraph

logger = structlog.get_logger(__name__)


def tokenize_text(text: str) -> list[str]:
    """Helper to tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


class HybridSearch:
    """Combines vector search, BM25, and graph entities via RRF."""

    def __init__(self, vector_store: VectorStore, entity_graph: EntityGraph):
        self.vector_store = vector_store
        self.entity_graph = entity_graph

    def _get_graph_boosted_chunks(self, entities: list[str], session_id: str) -> set[str]:
        """Fetch chunk IDs associated with extracted query entities and relationships from Neo4j."""
        boosted_chunks: set[str] = set()
        if not entities:
            return boosted_chunks

        logger.debug("graph_boost_lookup_start", entities=entities, session_id=session_id)
        
        for ent_name in entities:
            # 1. Substring search for matching entities in Neo4j within the session
            matched_nodes = self.entity_graph.search_entities(ent_name, session_id)
            for node in matched_nodes:
                c_id = node.get("chunk_id")
                if c_id:
                    boosted_chunks.add(c_id)

                # 2. Get relationships connected to the matched entity
                node_id = node.get("id")
                if node_id:
                    neighbors = self.entity_graph.get_neighbors_by_entity_id(node_id)
                    for rel in neighbors:
                        rel_chunk_id = rel.get("chunk_id")
                        if rel_chunk_id:
                            boosted_chunks.add(rel_chunk_id)

        logger.debug(
            "graph_boost_lookup_complete",
            session_id=session_id,
            num_boosted_chunks=len(boosted_chunks),
        )
        return boosted_chunks

    async def search(
        self,
        query: str,
        query_vector: list[float],
        session_id: str,
        extracted_entities: Optional[list[str]] = None,
        top_k: int = settings.hybrid_search_top_k,
    ) -> list[RetrievalResult]:
        """Perform hybrid retrieval: Vector + BM25 + Graph Boost fused via RRF.

        Args:
            query: The text query.
            query_vector: Dense embedding vector for the query.
            session_id: The session identifier.
            extracted_entities: Entities extracted from the query for graph boost.
            top_k: Number of hybrid results to return.

        Returns:
            List of fused RetrievalResult items.
        """
        logger.info("hybrid_search_start", query=query, session_id=session_id, top_k=top_k)

        # ─── 1. Vector Search ───
        # Retrieve slightly more than top_k to allow robust RRF fusion
        vector_candidates = self.vector_store.query_similarity(
            query_vector=query_vector,
            session_id=session_id,
            top_k=max(top_k * 2, 40),
        )
        
        # Create map of vector chunk_id -> RetrievalResult and ranking dictionary
        vector_rank: dict[str, int] = {}
        candidate_details: dict[str, dict[str, Any]] = {}
        
        for idx, res in enumerate(vector_candidates):
            vector_rank[res.chunk_id] = idx + 1
            candidate_details[res.chunk_id] = {
                "parent_id": res.parent_id,
                "text": res.text,
                "metadata": res.metadata,
            }

        # ─── 2. Sparse BM25 Search ───
        # Retrieve all session chunks from ChromaDB
        session_chunks = self.vector_store.get_session_chunks(session_id)
        bm25_candidates: list[tuple[str, float]] = []
        
        if session_chunks:
            # Build index on-the-fly
            doc_texts = [c["text"] for c in session_chunks]
            tokenized_corpus = [tokenize_text(text) for text in doc_texts]
            bm25_model = BM25Okapi(tokenized_corpus)
            
            tokenized_query = tokenize_text(query)
            scores = bm25_model.get_scores(tokenized_query)
            
            # Pair scores with chunk details and sort
            chunk_scores = []
            for i, chunk in enumerate(session_chunks):
                chunk_scores.append((chunk["id"], float(scores[i]), chunk))
                
            # Sort by score descending
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Map BM25 ranking
            bm25_rank: dict[str, int] = {}
            for rank_idx, (c_id, score, chunk) in enumerate(chunk_scores):
                # Only rank chunks that have a positive match score to prevent noise
                if score > 0.0:
                    bm25_rank[c_id] = rank_idx + 1
                    if c_id not in candidate_details:
                        meta = chunk.get("metadata") or {}
                        candidate_details[c_id] = {
                            "parent_id": meta.get("parent_id"),
                            "text": chunk["text"],
                            "metadata": meta,
                        }
        else:
            bm25_rank = {}

        # ─── 3. Graph Boost Lookup ───
        boosted_chunks = self._get_graph_boosted_chunks(extracted_entities or [], session_id)

        # ─── 4. Reciprocal Rank Fusion (RRF) ───
        rrf_scores: dict[str, float] = {}
        rrf_k = settings.rrf_k
        vec_w = settings.vector_weight
        bm25_w = settings.bm25_weight
        boost_w = settings.graph_boost_weight

        # Union of all candidates seen in vector or BM25
        all_chunk_ids = set(vector_rank.keys()).union(set(bm25_rank.keys()))

        for chunk_id in all_chunk_ids:
            score = 0.0
            
            # Vector contribution
            if chunk_id in vector_rank:
                score += vec_w / (rrf_k + vector_rank[chunk_id])
                
            # BM25 contribution
            if chunk_id in bm25_rank:
                score += bm25_w / (rrf_k + bm25_rank[chunk_id])
                
            # Graph boost contribution
            if chunk_id in boosted_chunks:
                score += boost_w
                
            rrf_scores[chunk_id] = score

        # ─── 5. Rank and Filter ───
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunks = sorted_chunks[:top_k]

        results: list[RetrievalResult] = []
        for chunk_id, score in top_chunks:
            details = candidate_details[chunk_id]
            
            # Determine primary source for reporting
            in_vec = chunk_id in vector_rank
            in_bm25 = chunk_id in bm25_rank
            in_graph = chunk_id in boosted_chunks
            
            if in_vec and in_bm25:
                source = "hybrid"
            elif in_vec:
                source = "vector"
            elif in_bm25:
                source = "bm25"
            else:
                source = "graph"
                
            if in_graph:
                source += "+graph"

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    parent_id=details["parent_id"],
                    text=details["text"],
                    score=round(score, 6),
                    source=source,
                    metadata=details["metadata"],
                )
            )

        logger.info(
            "hybrid_search_complete",
            session_id=session_id,
            results_found=len(results),
            top_score=results[0].score if results else 0.0,
        )
        return results
