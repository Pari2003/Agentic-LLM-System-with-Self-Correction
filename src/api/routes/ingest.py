"""
PDF ingestion routes.

Endpoints:
- POST /sessions/{session_id}/ingest — Upload and ingest a PDF document
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.dependencies import get_doc_store, get_vector_store, get_graph_store, get_llm_client
from src.ingestion.embedder import Embedder
from src.ingestion.image_captioner import ImageCaptioner
from src.ingestion.parent_child_linker import ParentChildLinker
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.semantic_chunker import SemanticChunker
from src.models.llm_client import OllamaClient
from src.models.schemas import Document
from src.storage.document_store import DocumentStore
from src.storage.entity_graph import EntityGraph
from src.storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Ingestion"])


# ─── Response Models ─────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    document_id: str
    filename: str
    title: str | None = None
    total_pages: int = 0
    total_parent_chunks: int = 0
    total_child_chunks: int = 0
    total_tables: int = 0
    total_figures: int = 0
    session_id: str
    message: str


# ─── Routes ───────────────────────────────────────────────────────────────

@router.post(
    "/sessions/{session_id}/ingest",
    response_model=IngestResponse,
    status_code=201,
    summary="Ingest PDF Document",
    description="Upload a PDF document for ingestion. The document will be parsed, chunked semantically, embedded, and stored across ChromaDB, SQLite, and Neo4j.",
)
async def ingest_pdf(
    session_id: str,
    file: UploadFile = File(..., description="PDF file to ingest"),
    doc_store: DocumentStore = Depends(get_doc_store),
    vector_store: VectorStore = Depends(get_vector_store),
    graph_store: EntityGraph = Depends(get_graph_store),
    llm_client: OllamaClient = Depends(get_llm_client),
):
    """Ingest a PDF document through the full pipeline.

    Pipeline stages:
    1. PDF Parsing (text, tables, images)
    2. Semantic Chunking
    3. Parent-Child Linking
    4. Embedding Generation
    5. Multi-tier Storage
    """
    # Validate session exists
    session = doc_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    logger.info("ingestion_start", filename=file.filename, session_id=session_id)

    try:
        # Save uploaded file temporarily
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # 1. Parse PDF
        parser = PDFParser()
        parsed = parser.parse(str(tmp_path))

        # 2. Semantic Chunking
        chunker = SemanticChunker(llm_client)
        raw_chunks = await chunker.chunk(parsed["text_blocks"])

        # 3. Parent-Child Linking
        linker = ParentChildLinker()
        parent_chunks, child_chunks = linker.link(
            raw_chunks=raw_chunks,
            document_id="pending",  # Will be set after document creation
            session_id=session_id,
            section_titles=parsed.get("sections", {}),
            page_numbers=parsed.get("pages", {}),
        )

        # 4. Create Document record
        doc = Document(
            filename=file.filename,
            title=parsed.get("title"),
            authors=parsed.get("authors"),
            abstract=parsed.get("abstract"),
            total_pages=parsed.get("total_pages", 0),
            total_chunks=len(parent_chunks) + len(child_chunks),
            total_parent_chunks=len(parent_chunks),
            total_child_chunks=len(child_chunks),
            total_tables=len(parsed.get("tables", [])),
            total_figures=len(parsed.get("images", [])),
            session_id=session_id,
        )

        # Update document_id in chunks
        for pc in parent_chunks:
            pc.document_id = doc.id
        for cc in child_chunks:
            cc.metadata.document_id = doc.id

        # 5. Embed child chunks
        embedder = Embedder(llm_client)
        child_chunks = await embedder.embed_chunks(child_chunks)

        # 6. Store everything
        doc_store.save_document(doc)
        doc_store.save_parent_chunks(parent_chunks)

        child_ids = [c.id for c in child_chunks]
        parent_ids = [c.metadata.parent_id or "" for c in child_chunks]
        types = [c.metadata.chunk_type.value for c in child_chunks]
        pages = [c.metadata.page_number or 0 for c in child_chunks]
        sections = [c.metadata.section_title or "" for c in child_chunks]
        indexes = [c.metadata.chunk_index for c in child_chunks]

        doc_store.save_child_metadata_batch(
            child_ids=child_ids,
            parent_ids=parent_ids,
            doc_id=doc.id,
            session_id=session_id,
            types=types,
            pages=pages,
            sections=sections,
            indexes=indexes,
        )

        vector_store.add_chunks(child_chunks)

        # 7. Extract and store entities (if graph store is available)
        try:
            entities, relationships = await _extract_entities(
                parent_chunks, doc.id, llm_client
            )
            if entities:
                graph_store.save_graph_elements(entities, relationships)
        except Exception as e:
            logger.warning("entity_extraction_skipped", error=str(e))

        # Update session document count
        doc_store.increment_session_doc_count(session_id)

        # Cleanup temp file
        tmp_path.unlink(missing_ok=True)

        logger.info(
            "ingestion_complete",
            document_id=doc.id,
            filename=file.filename,
            parent_chunks=len(parent_chunks),
            child_chunks=len(child_chunks),
        )

        return IngestResponse(
            document_id=doc.id,
            filename=file.filename,
            title=doc.title,
            total_pages=doc.total_pages,
            total_parent_chunks=doc.total_parent_chunks,
            total_child_chunks=doc.total_child_chunks,
            total_tables=doc.total_tables,
            total_figures=doc.total_figures,
            session_id=session_id,
            message=f"Successfully ingested '{file.filename}' with {len(parent_chunks)} parent and {len(child_chunks)} child chunks.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("ingestion_error", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


async def _extract_entities(parent_chunks, doc_id, llm_client):
    """Extract entities and relationships from parent chunks using LLM."""
    from src.models.schemas import Entity, EntityType, Relationship

    all_entities = []
    all_relationships = []

    for pc in parent_chunks:
        system_prompt = (
            "Extract entities and relationships from this text. "
            "Entities: [name, type] where type is one of: METHOD, DATASET, METRIC, MODEL, PERSON, ORGANIZATION, TASK. "
            "Relationships: [source, relation, target] where relation is one of: USES, EVALUATED_ON, OUTPERFORMS, INTRODUCED, PART_OF. "
            "Output JSON: {\"entities\": [...], \"relationships\": [...]}"
        )
        try:
            result = await llm_client.generate_json(
                prompt=pc.text[:2000],
                system_prompt=system_prompt,
                temperature=0.0,
            )

            for e in result.get("entities", []):
                name = e.get("name", "").strip()
                etype = e.get("type", "METHOD").upper()
                if name and hasattr(EntityType, etype):
                    all_entities.append(Entity(
                        name=name,
                        entity_type=EntityType(etype.lower()),
                        document_id=doc_id,
                        chunk_id=pc.id,
                    ))

            for r in result.get("relationships", []):
                source = r.get("source", "").strip()
                target = r.get("target", "").strip()
                rel_type = r.get("relation", "USES").upper()
                if source and target:
                    all_relationships.append(Relationship(
                        source_entity_id="",
                        source_entity_name=source,
                        relation_type=rel_type,
                        target_entity_id="",
                        target_entity_name=target,
                        chunk_id=pc.id,
                    ))
        except Exception:
            continue

    return all_entities, all_relationships
