"""
Session Lifecycle Manager.

Coordinates session creation, access validation, and TTL-based background cleanup
across all three storage layers:
1. SQLite metadata store (sessions, documents, parent/child mapping tables)
2. SQLite knowledge graph store (entities, relationships)
3. ChromaDB vector database (removes session-tagged embeddings)

Uses APScheduler to run a background cleanup task periodically.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from apscheduler.schedulers.background import BackgroundScheduler

from src.config import settings
from src.models.schemas import Session
from src.storage.document_store import DocumentStore
from src.storage.entity_graph import EntityGraph
from src.storage.vector_store import VectorStore

logger = structlog.get_logger(__name__)


class SessionManager:
    """Manages session creation, TTL updates, and background cleanup jobs."""

    def __init__(
        self,
        doc_store: DocumentStore,
        vector_store: VectorStore,
        graph_store: EntityGraph,
        ttl_hours: int = settings.session_ttl_hours,
        cleanup_interval_mins: int = settings.cleanup_interval_minutes,
        start_scheduler: bool = True,
    ):
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.ttl_hours = ttl_hours
        self.cleanup_interval_mins = cleanup_interval_mins
        
        self.scheduler: Optional[BackgroundScheduler] = None
        if start_scheduler:
            self._start_cleanup_scheduler()

    def _start_cleanup_scheduler(self) -> None:
        """Start the background scheduler to periodically purge expired sessions."""
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.cleanup_expired_sessions,
            "interval",
            minutes=self.cleanup_interval_mins,
            id="session_cleanup_job",
        )
        self.scheduler.start()
        logger.info(
            "session_cleanup_scheduler_started",
            interval_minutes=self.cleanup_interval_mins,
        )

    def shutdown(self) -> None:
        """Stop the background scheduler gracefully."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("session_cleanup_scheduler_shutdown")

    # ─── Lifecycle Actions ────────────────────────────────────────────────

    def create_session(self) -> Session:
        """Create a new session with an expiry time based on TTL config."""
        session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{timedelta(microseconds=1)}"
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(hours=self.ttl_hours)

        session = Session(
            id=session_id,
            created_at=created_at,
            expires_at=expires_at,
            document_count=0,
            query_count=0,
            is_active=True,
        )

        self.doc_store.save_session(session)
        logger.info(
            "session_created",
            session_id=session.id,
            expires_at=session.expires_at.isoformat(),
        )
        return session

    def validate_session(self, session_id: str) -> bool:
        """Verify if a session exists, is active, and has not expired.

        Side effect: Extends the expiration time (sliding window expiry).
        """
        session = self.doc_store.get_session(session_id)
        if not session:
            logger.warning("session_validation_failed_not_found", session_id=session_id)
            return False

        if not session.is_active:
            logger.warning("session_validation_failed_inactive", session_id=session_id)
            return False

        now = datetime.now(timezone.utc)
        if session.expires_at and session.expires_at < now:
            logger.info("session_validation_failed_expired", session_id=session_id)
            # Cleanup immediately
            self.close_session(session_id)
            return False

        # Session is valid: Extend TTL (sliding window expiry)
        session.expires_at = now + timedelta(hours=self.ttl_hours)
        self.doc_store.save_session(session)
        logger.debug("session_ttl_extended", session_id=session_id, new_expiry=session.expires_at.isoformat())
        return True

    def close_session(self, session_id: str) -> None:
        """Explicitly wipe and close a session."""
        logger.info("session_close_start", session_id=session_id)
        
        # 1. Clean up ChromaDB collection vectors tagged with this session
        try:
            self.vector_store.delete_session_chunks(session_id)
        except Exception as e:
            logger.error("session_cleanup_chromadb_error", session_id=session_id, error=str(e))

        # 2. Delete metadata in SQLite (cascades deletes to documents, parent_chunks, metadata)
        try:
            self.doc_store.delete_session(session_id)
        except Exception as e:
            logger.error("session_cleanup_sqlite_error", session_id=session_id, error=str(e))

        # 3. Clean up Neo4j knowledge graph session data
        try:
            self.graph_store.delete_session_graph(session_id)
        except Exception as e:
            logger.error("session_cleanup_neo4j_error", session_id=session_id, error=str(e))

        logger.info("session_close_complete", session_id=session_id)

    def cleanup_expired_sessions(self) -> None:
        """Triggered periodically by background scheduler to wipe expired data."""
        logger.info("background_session_cleanup_job_start")
        try:
            expired_ids = self.doc_store.get_expired_sessions()
            if expired_ids:
                logger.info("expired_sessions_detected", count=len(expired_ids), ids=expired_ids)
                for session_id in expired_ids:
                    self.close_session(session_id)
            else:
                logger.info("no_expired_sessions_found")
        except Exception as e:
            logger.error("background_session_cleanup_failed", error=str(e))
