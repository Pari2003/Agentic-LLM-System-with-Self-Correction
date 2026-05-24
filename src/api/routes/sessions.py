"""
Session management routes.

Endpoints:
- POST   /sessions          — Create a new session
- GET    /sessions/{id}     — Get session details
- DELETE /sessions/{id}     — Delete session and all associated data
- GET    /sessions          — List all active sessions
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_doc_store, get_vector_store, get_graph_store
from src.models.schemas import Session
from src.storage.document_store import DocumentStore
from src.storage.entity_graph import EntityGraph
from src.storage.vector_store import VectorStore

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# ─── Request / Response Models ────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    """Request body for creating a new session."""
    ttl_hours: int = Field(default=24, ge=1, le=168, description="Session time-to-live in hours")


class SessionResponse(BaseModel):
    """Response model for session operations."""
    id: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    document_count: int = 0
    query_count: int = 0
    is_active: bool = True


class SessionDeleteResponse(BaseModel):
    """Response model for session deletion."""
    id: str
    status: str = "deleted"
    message: str


# ─── Routes ───────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=SessionResponse,
    status_code=201,
    summary="Create Session",
    description="Create a new user session with configurable TTL. All ingested documents and queries will be scoped to this session.",
)
async def create_session(
    request: CreateSessionRequest = CreateSessionRequest(),
    doc_store: DocumentStore = Depends(get_doc_store),
):
    """Create a new session with a unique ID and TTL."""
    now = datetime.now(timezone.utc)
    session = Session(
        created_at=now,
        expires_at=now + timedelta(hours=request.ttl_hours),
        is_active=True,
    )
    doc_store.save_session(session)

    return SessionResponse(
        id=session.id,
        created_at=session.created_at,
        expires_at=session.expires_at,
        document_count=session.document_count,
        query_count=session.query_count,
        is_active=session.is_active,
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get Session",
    description="Retrieve details of a specific session by ID.",
)
async def get_session(
    session_id: str,
    doc_store: DocumentStore = Depends(get_doc_store),
):
    """Get session details by ID."""
    session = doc_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return SessionResponse(
        id=session.id,
        created_at=session.created_at,
        expires_at=session.expires_at,
        document_count=session.document_count,
        query_count=session.query_count,
        is_active=session.is_active,
    )


@router.delete(
    "/{session_id}",
    response_model=SessionDeleteResponse,
    summary="Delete Session",
    description="Delete a session and all associated data (vectors, documents, entities).",
)
async def delete_session(
    session_id: str,
    doc_store: DocumentStore = Depends(get_doc_store),
    vector_store: VectorStore = Depends(get_vector_store),
    graph_store: EntityGraph = Depends(get_graph_store),
):
    """Delete session and cascade-clean all associated data."""
    session = doc_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    # Clean up all storage layers
    try:
        vector_store.delete_by_session(session_id)
    except Exception:
        pass  # ChromaDB may not have data for this session

    try:
        graph_store.delete_session_graph(session_id)
    except Exception:
        pass  # Neo4j may not have data for this session

    doc_store.delete_session(session_id)

    return SessionDeleteResponse(
        id=session_id,
        status="deleted",
        message=f"Session '{session_id}' and all associated data have been deleted.",
    )


@router.get(
    "",
    response_model=list[SessionResponse],
    summary="List Sessions",
    description="List all active sessions.",
)
async def list_sessions(
    doc_store: DocumentStore = Depends(get_doc_store),
):
    """List all active sessions."""
    sessions = doc_store.list_sessions()
    return [
        SessionResponse(
            id=s.id,
            created_at=s.created_at,
            expires_at=s.expires_at,
            document_count=s.document_count,
            query_count=s.query_count,
            is_active=s.is_active,
        )
        for s in sessions
    ]
