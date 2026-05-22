"""
Pydantic v2 data models for the entire agentic RAG pipeline.

This module defines the type-safe data structures used across all components:
- Document ingestion (Document, Chunk, ParentChunk)
- Entity graph (Entity, Relationship)
- Retrieval (RetrievalResult, RankedContext, QueryAnalysis)
- Hallucination detection (Claim, ClaimVerification, CritiqueResult)
- API request/response (QueryRequest, QueryResponse, Citation)
- Evaluation (RetrievalEvalResult, GenerationEvalResult, EvalResult)
- Session management (Session)
- Metrics (PipelineMetrics)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class ChunkType(str, Enum):
    """Type of content a chunk contains."""

    TEXT = "text"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"


class ChunkLevel(str, Enum):
    """Hierarchical level of a chunk in the parent-child structure."""

    CHILD = "child"    # Small (~256 tok), embedded in ChromaDB, used for retrieval
    PARENT = "parent"  # Large (~1024 tok), stored in SQLite, used for generation


class EntityType(str, Enum):
    """Types of entities extracted from research papers."""

    METHOD = "method"
    DATASET = "dataset"
    METRIC = "metric"
    MODEL = "model"
    PERSON = "person"
    ORGANIZATION = "organization"
    TASK = "task"


class EntailmentResult(str, Enum):
    """NLI-style entailment classification for hallucination detection."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class QueryComplexity(str, Enum):
    """Complexity classification for incoming queries."""

    SIMPLE = "simple"      # Single-hop factual: "What is the learning rate?"
    MODERATE = "moderate"   # Multi-fact: "Compare accuracy and F1 scores"
    COMPLEX = "complex"     # Multi-hop reasoning: "Why did method A outperform B on dataset C?"


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT & CHUNK MODELS
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_id() -> str:
    """Generate a unique ID string."""
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    """Generate current UTC timestamp."""
    return datetime.now(timezone.utc)


class Document(BaseModel):
    """Represents an ingested PDF document with extracted metadata."""

    id: str = Field(default_factory=_generate_id)
    filename: str
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    abstract: Optional[str] = None
    summary: Optional[str] = None
    total_pages: int = 0
    total_chunks: int = 0
    total_parent_chunks: int = 0
    total_child_chunks: int = 0
    total_tables: int = 0
    total_figures: int = 0
    session_id: str
    ingested_at: datetime = Field(default_factory=_utc_now)


class ChunkMetadata(BaseModel):
    """Metadata attached to each chunk for filtering and traceability."""

    document_id: str
    session_id: str
    chunk_type: ChunkType = ChunkType.TEXT
    chunk_level: ChunkLevel = ChunkLevel.CHILD
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    page_numbers: list[int] = []
    parent_id: Optional[str] = None  # For child chunks → points to parent
    chunk_index: int = 0             # Position within the document

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Convert to flat dict compatible with ChromaDB metadata storage.

        ChromaDB metadata only supports str, int, float, bool values.
        """
        return {
            "document_id": self.document_id,
            "session_id": self.session_id,
            "chunk_type": self.chunk_type.value,
            "chunk_level": self.chunk_level.value,
            "section_title": self.section_title or "",
            "page_number": self.page_number or 0,
            "parent_id": self.parent_id or "",
            "chunk_index": self.chunk_index,
        }


class Chunk(BaseModel):
    """A text chunk from a document — the fundamental unit of the system.

    Child chunks (~256 tokens) are embedded and stored in ChromaDB for retrieval.
    Their embeddings are compared against query embeddings during vector search.
    """

    id: str = Field(default_factory=_generate_id)
    text: str
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = None
    token_count: int = 0


class ParentChunk(BaseModel):
    """Large chunk (~1024 tokens) stored in SQLite for generation context.

    When a child chunk is retrieved, we look up its parent to provide the LLM
    with richer context for answer generation. This is the 'small-to-big' pattern.
    """

    id: str = Field(default_factory=_generate_id)
    text: str
    document_id: str
    session_id: str
    section_title: Optional[str] = None
    page_numbers: list[int] = []
    child_ids: list[str] = []
    token_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY GRAPH MODELS (Lightweight GraphRAG)
# ═══════════════════════════════════════════════════════════════════════════════


class Entity(BaseModel):
    """An entity extracted from a chunk (method, dataset, model, person, etc.).

    Entities are stored in SQLite with FTS5 indexing for fast lookup.
    They enable entity-aware retrieval: queries mentioning known entities
    get a retrieval boost for chunks containing those entities.
    """

    id: str = Field(default_factory=_generate_id)
    name: str
    entity_type: EntityType
    document_id: str
    chunk_id: str


class Relationship(BaseModel):
    """A directed relationship between two entities.

    Example: Entity("Transformer", MODEL) --EVALUATED_ON--> Entity("WMT 2014", DATASET)

    Relationships enable multi-hop retrieval: "What datasets were used to evaluate
    transformer models?" → graph traversal finds connected entities.
    """

    id: str = Field(default_factory=_generate_id)
    source_entity_id: str
    source_entity_name: str
    relation_type: str  # INTRODUCED, USES, EVALUATED_ON, OUTPERFORMS, PART_OF, etc.
    target_entity_id: str
    target_entity_name: str
    chunk_id: str
    confidence: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class RetrievalResult(BaseModel):
    """A single result from Stage 1 hybrid search (before parent expansion)."""

    chunk_id: str
    parent_id: Optional[str] = None
    text: str
    score: float
    source: str  # "vector", "bm25", "graph"
    metadata: dict[str, Any] = {}


class RankedContext(BaseModel):
    """A parent chunk with its final ranking score after all 3 retrieval stages.

    This is what gets passed to the Generator agent as context.
    """

    parent_chunk: ParentChunk
    rrf_score: float = 0.0
    llm_relevance_score: Optional[float] = None
    contributing_child_ids: list[str] = []  # Which child chunks contributed
    final_rank: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class QueryAnalysis(BaseModel):
    """Output of the Query Analyzer agent — structured understanding of the query.

    The analyzer classifies complexity, extracts entities for graph lookup,
    and generates sub-queries for multi-hop questions.
    """

    original_query: str
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    sub_queries: list[str] = []
    extracted_entities: list[str] = []
    search_queries: list[str] = []  # Rewritten queries optimized for retrieval
    intent: str = ""                # Brief description of what the user is asking


class QueryRequest(BaseModel):
    """Incoming query from the API — what the user submits."""

    question: str
    session_id: str
    max_sources: int = Field(default=5, ge=1, le=20)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_self_correction: bool = True
    max_correction_iterations: int = Field(default=2, ge=0, le=5)


# ═══════════════════════════════════════════════════════════════════════════════
# HALLUCINATION DETECTION MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class Claim(BaseModel):
    """An individual factual claim extracted from the LLM's response.

    The Critic agent decomposes the response into atomic claims,
    then verifies each claim against the source documents.
    """

    id: str = Field(default_factory=_generate_id)
    text: str
    source_sentence: str  # The original sentence this claim was extracted from


class ClaimVerification(BaseModel):
    """Verification result for a single claim using the 3-layer check.

    Layer 1 (embedding_similarity): cosine similarity between claim and source chunks
    Layer 2 (entailment): LLM-as-NLI — does the source support/contradict/neutral?
    Layer 3 (keyword_overlap): named entities and numbers overlap check
    """

    claim: Claim
    # Layer 1: Embedding similarity
    embedding_similarity: float = 0.0
    best_matching_chunk_id: Optional[str] = None
    best_matching_text: Optional[str] = None
    # Layer 2: NLI entailment
    entailment_result: EntailmentResult = EntailmentResult.NEUTRAL
    entailment_score: float = 0.0
    # Layer 3: Keyword overlap
    keyword_overlap_score: float = 0.0
    matched_keywords: list[str] = []
    missing_keywords: list[str] = []
    # Aggregated result
    overall_confidence: float = 0.0
    is_hallucination: bool = False
    explanation: Optional[str] = None


class CritiqueResult(BaseModel):
    """Complete output of the Critic agent — assessment of all claims."""

    claims: list[ClaimVerification] = []
    overall_confidence: float = 0.0
    hallucinated_claims: list[ClaimVerification] = []
    supported_claims: list[ClaimVerification] = []
    total_claims: int = 0
    hallucination_rate: float = 0.0
    needs_refinement: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class Citation(BaseModel):
    """A citation reference in the generated response.

    Maps [1], [2], etc. in the response text to actual source documents.
    """

    source_id: int                        # [1], [2], etc.
    chunk_id: str
    document_id: str
    document_title: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    relevant_text: str                    # The source text snippet


class PipelineMetrics(BaseModel):
    """Latency and resource metrics for a single query through the pipeline.

    Tracks time spent in each agent for performance analysis and bottleneck detection.
    """

    query_analysis_ms: float = 0.0
    metadata_filter_ms: float = 0.0
    hybrid_search_ms: float = 0.0
    parent_expansion_ms: float = 0.0
    llm_rerank_ms: float = 0.0
    generation_ms: float = 0.0
    critique_ms: float = 0.0
    refinement_ms: float = 0.0
    total_ms: float = 0.0
    correction_iterations: int = 0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    retrieval_sources_used: int = 0


class QueryResponse(BaseModel):
    """Full response from the agentic pipeline — returned to the user.

    Contains the answer, citations, confidence score, detailed claim report,
    query analysis, and performance metrics.
    """

    answer: str
    citations: list[Citation] = []
    confidence_score: float
    claim_report: CritiqueResult
    query_analysis: QueryAnalysis
    metrics: PipelineMetrics
    correction_iterations: int = 0
    session_id: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class Session(BaseModel):
    """A user session with session-scoped vector storage.

    Vector data in ChromaDB is tagged with session_id and automatically
    cleaned up after TTL expiry. Metadata in SQLite persists across sessions.
    """

    id: str = Field(default_factory=_generate_id)
    created_at: datetime = Field(default_factory=_utc_now)
    expires_at: Optional[datetime] = None
    document_count: int = 0
    query_count: int = 0
    is_active: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class RetrievalEvalResult(BaseModel):
    """Retrieval evaluation metrics — computed without LLM calls (fast, free).

    These metrics measure how well the retrieval pipeline finds relevant contexts.
    """

    context_precision: float = 0.0     # Relevant chunks ranked higher?
    context_recall: float = 0.0        # Did we find all relevant chunks?
    mrr_at_k: float = 0.0             # Where does first relevant result appear?
    ndcg_at_k: float = 0.0            # Quality of ranking order
    hit_rate_at_k: float = 0.0        # Does ANY relevant chunk appear in top-k?
    k: int = 5


class GenerationEvalResult(BaseModel):
    """Generation evaluation metrics — requires LLM-as-Judge calls.

    Only run after retrieval metrics are satisfactory (cost optimization).
    """

    faithfulness: float = 0.0          # Is every claim grounded in sources?
    answer_relevancy: float = 0.0      # Does the answer address the question?
    completeness: float = 0.0          # Does the answer cover all aspects?


class EvalResult(BaseModel):
    """Combined evaluation result for a single query — used in benchmarks."""

    query: str
    retrieval: RetrievalEvalResult = Field(default_factory=RetrievalEvalResult)
    generation: GenerationEvalResult = Field(default_factory=GenerationEvalResult)
    ground_truth_answer: Optional[str] = None
    predicted_answer: str = ""
    pipeline_metrics: PipelineMetrics = Field(default_factory=PipelineMetrics)
