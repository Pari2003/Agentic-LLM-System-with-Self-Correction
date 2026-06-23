"""
SQLite Document Store Adapter.

Responsible for persistent metadata storage:
1. Document registries (filenames, page counts, title, authors)
2. Parent chunks (full text of 1024-token parent blocks)
3. Session lifecycles (creation, activity, expiry times)
4. Child chunk lookups (mapping child IDs to parents and page numbers)

Design choices:
- SQLite: Single file, fast, embedded, async-friendly via standard library, perfect for lightweight persistence.
- Table initialization on module import/creation.
- Safe serialization/deserialization of lists (page numbers, child IDs) to JSON strings in DB columns.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from src.config import settings
from src.models.schemas import Document, ParentChunk, Session

logger = structlog.get_logger(__name__)


class DocumentStore:
    """Manages document, parent chunk, and session records in SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.sqlite_file
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL enabled for concurrent reads."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable write-ahead logging (WAL) for better concurrent read/write performance
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        """Initialize tables if they do not exist."""
        logger.info("sqlite_db_init_start", path=str(self.db_path))
        with self._get_connection() as conn:
            # 1. Sessions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    is_active INTEGER DEFAULT 1
                )
                """
            )

            # 2. Documents table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    title TEXT,
                    authors TEXT, -- JSON array of strings
                    abstract TEXT,
                    summary TEXT,
                    total_pages INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    total_parent_chunks INTEGER DEFAULT 0,
                    total_child_chunks INTEGER DEFAULT 0,
                    total_tables INTEGER DEFAULT 0,
                    total_figures INTEGER DEFAULT 0,
                    session_id TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                )
                """
            )

            # 3. Parent Chunks table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    section_title TEXT,
                    page_numbers TEXT, -- JSON array of integers
                    token_count INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                )
                """
            )

            # 4. Child-to-Parent Lookup index table (for fast hybrid retrieval routing)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS child_chunk_metadata (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    section_title TEXT,
                    chunk_index INTEGER NOT NULL,
                    FOREIGN KEY (parent_id) REFERENCES parent_chunks (id) ON DELETE CASCADE,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                )
                """
            )

            # Indexes for faster queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_session ON documents(session_id);")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_parents_doc ON parent_chunks(document_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_parents_session ON parent_chunks(session_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_child_meta_parent ON child_chunk_metadata(parent_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_child_meta_session ON child_chunk_metadata(session_id);"
            )

            conn.commit()

        logger.info("sqlite_db_init_complete")

    # ─── Session Management ───────────────────────────────────────────────

    def save_session(self, session: Session) -> None:
        """Create or update a session in SQLite."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, created_at, expires_at, is_active)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    expires_at=excluded.expires_at,
                    is_active=excluded.is_active
                """,
                (
                    session.id,
                    session.created_at.isoformat(),
                    session.expires_at.isoformat() if session.expires_at else None,
                    1 if session.is_active else 0,
                ),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                return None
            return Session(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
                is_active=bool(row["is_active"]),
            )

    def delete_session(self, session_id: str) -> None:
        """Wipe all database records related to a session (cascades automatically)."""
        logger.info("sqlite_session_delete_start", session_id=session_id)
        with self._get_connection() as conn:
            # ON DELETE CASCADE automatically deletes corresponding documents, parent_chunks, and child_metadata!
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
        logger.info("sqlite_session_delete_complete", session_id=session_id)

    def list_sessions(self) -> list[Session]:
        """List all active sessions."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM sessions").fetchall()
            return [
                Session(
                    id=row["id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    expires_at=datetime.fromisoformat(row["expires_at"])
                    if row["expires_at"]
                    else None,
                    is_active=bool(row["is_active"]),
                )
                for row in rows
            ]

    # ─── Document Management ──────────────────────────────────────────────

    def save_document(self, doc: Document) -> None:
        """Save a new document metadata record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, filename, title, authors, abstract, summary,
                    total_pages, total_chunks, total_parent_chunks,
                    total_child_chunks, total_tables, total_figures,
                    session_id, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.id,
                    doc.filename,
                    doc.title,
                    json.dumps(doc.authors or []),
                    doc.abstract,
                    doc.summary,
                    doc.total_pages,
                    doc.total_chunks,
                    doc.total_parent_chunks,
                    doc.total_child_chunks,
                    doc.total_tables,
                    doc.total_figures,
                    doc.session_id,
                    doc.ingested_at.isoformat(),
                ),
            )
            conn.commit()

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Fetch document metadata by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if not row:
                return None
            return Document(
                id=row["id"],
                filename=row["filename"],
                title=row["title"],
                authors=json.loads(row["authors"]) if row["authors"] else [],
                abstract=row["abstract"],
                summary=row["summary"],
                total_pages=row["total_pages"],
                total_chunks=row["total_chunks"],
                total_parent_chunks=row["total_parent_chunks"],
                total_child_chunks=row["total_child_chunks"],
                total_tables=row["total_tables"],
                total_figures=row["total_figures"],
                session_id=row["session_id"],
                ingested_at=datetime.fromisoformat(row["ingested_at"]),
            )

    def get_documents_by_session(self, session_id: str) -> list[Document]:
        """Get all document records belonging to a session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE session_id = ?", (session_id,)
            ).fetchall()
            docs = []
            for row in rows:
                docs.append(
                    Document(
                        id=row["id"],
                        filename=row["filename"],
                        title=row["title"],
                        authors=json.loads(row["authors"]) if row["authors"] else [],
                        abstract=row["abstract"],
                        summary=row["summary"],
                        total_pages=row["total_pages"],
                        total_chunks=row["total_chunks"],
                        total_parent_chunks=row["total_parent_chunks"],
                        total_child_chunks=row["total_child_chunks"],
                        total_tables=row["total_tables"],
                        total_figures=row["total_figures"],
                        session_id=row["session_id"],
                        ingested_at=datetime.fromisoformat(row["ingested_at"]),
                    )
                )
            return docs

    # ─── Parent Chunk Management ──────────────────────────────────────────

    def save_parent_chunks(self, parents: list[ParentChunk]) -> None:
        """Bulk save Parent Chunks into SQLite."""
        if not parents:
            return

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO parent_chunks (
                    id, text, document_id, session_id, section_title, page_numbers, token_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        p.id,
                        p.text,
                        p.document_id,
                        p.session_id,
                        p.section_title,
                        json.dumps(p.page_numbers),
                        p.token_count,
                    )
                    for p in parents
                ],
            )
            conn.commit()

    def get_parent_chunk(self, parent_id: str) -> Optional[ParentChunk]:
        """Retrieve a specific parent chunk by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM parent_chunks WHERE id = ?", (parent_id,)).fetchone()
            if not row:
                return None

            # Find associated child ids from lookup
            child_rows = conn.execute(
                "SELECT id FROM child_chunk_metadata WHERE parent_id = ?", (parent_id,)
            ).fetchall()
            child_ids = [r["id"] for r in child_rows]

            return ParentChunk(
                id=row["id"],
                text=row["text"],
                document_id=row["document_id"],
                session_id=row["session_id"],
                section_title=row["section_title"],
                page_numbers=json.loads(row["page_numbers"]) if row["page_numbers"] else [],
                child_ids=child_ids,
                token_count=row["token_count"],
            )

    # ─── Child Chunk Metadata & Lookups ───────────────────────────────────

    def save_child_metadata_batch(
        self,
        child_ids: list[str],
        parent_ids: list[str],
        doc_id: str,
        session_id: str,
        types: list[str],
        pages: list[int],
        sections: list[Optional[str]],
        indexes: list[int],
    ) -> None:
        """Bulk save child chunk mappings to parent chunks."""
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO child_chunk_metadata (
                    id, parent_id, document_id, session_id, chunk_type, page_number, section_title, chunk_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                zip(
                    child_ids,
                    parent_ids,
                    [doc_id] * len(child_ids),
                    [session_id] * len(child_ids),
                    types,
                    pages,
                    sections,
                    indexes,
                ),
            )
            conn.commit()

    def get_parent_by_child_id(self, child_id: str) -> Optional[ParentChunk]:
        """Find the ParentChunk that contains the given child chunk ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT parent_id FROM child_chunk_metadata WHERE id = ?
                """,
                (child_id,),
            ).fetchone()
            if not row:
                return None
            return self.get_parent_chunk(row["parent_id"])
    
    # Existing Sessions
    def get_existing_sessions(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT DISTINCT
                s.id,
                d.filename,
                s.created_at
                FROM sessions s
                JOIN documents d
                    ON s.id = d.session_id
                ORDER BY s.created_at DESC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def get_expired_sessions(self) -> list[str]:
        """Find all session IDs that have expired past their TTL time."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id FROM sessions WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)
            ).fetchall()
            return [r["id"] for r in rows]
