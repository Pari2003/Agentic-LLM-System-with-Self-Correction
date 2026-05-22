"""
SQLite Entity Graph Store (Lightweight GraphRAG Database).

Responsible for storing the knowledge graph extracted from chunks:
1. Entity nodes (name, type, source chunk/doc)
2. Directed relationships/edges (source, target, relation label, confidence)

Design choices:
- Store in the same SQLite database as document metadata to keep storage simple.
- Uses SQLite FTS5 (Full-Text Search) for rapid text matches on entities mentioned in user queries.
- Helps retrieve adjacent entities during Stage 1 hybrid retrieval.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

import structlog

from src.config import settings
from src.models.schemas import Entity, Relationship

logger = structlog.get_logger(__name__)


class EntityGraph:
    """Manages knowledge graph nodes and edges stored in SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.sqlite_file
        self._init_graph_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_graph_tables(self) -> None:
        """Create entity, relationship, and FTS tables."""
        logger.info("entity_graph_init_start", path=str(self.db_path))
        with self._get_connection() as conn:
            # 1. Entities Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
                """
            )

            # 2. Relationships Table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL,
                    source_entity_name TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    target_entity_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (source_entity_id) REFERENCES entities (id) ON DELETE CASCADE,
                    FOREIGN KEY (target_entity_id) REFERENCES entities (id) ON DELETE CASCADE
                )
                """
            )

            # 3. Create FTS5 Virtual Table for Entities (for fast entity matching)
            # FTS5 tables cannot have external primary keys or foreign keys directly
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
                    name,
                    entity_type,
                    content='entities',
                    content_rowid='rowid'
                )
                """
            )

            # Trigger to keep FTS table in sync on INSERT
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_entities_ai AFTER INSERT ON entities BEGIN
                    INSERT INTO entities_fts(rowid, name, entity_type) 
                    VALUES (new.rowid, new.name, new.entity_type);
                END
                """
            )

            # Trigger to keep FTS table in sync on DELETE
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_entities_ad AFTER DELETE ON entities BEGIN
                    INSERT INTO entities_fts(entities_fts, rowid, name, entity_type) 
                    VALUES('delete', old.rowid, old.name, old.entity_type);
                END
                """
            )

            # Indexes for graph lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities(chunk_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);")

            conn.commit()
        
        logger.info("entity_graph_init_complete")

    def save_graph_elements(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Bulk save entities and relationships into the graph store."""
        if not entities and not relationships:
            return

        with self._get_connection() as conn:
            # 1. Save Entities
            if entities:
                conn.executemany(
                    """
                    INSERT INTO entities (id, name, entity_type, document_id, chunk_id)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    [(e.id, e.name, e.entity_type.value, e.document_id, e.chunk_id) for e in entities],
                )

            # 2. Save Relationships
            if relationships:
                conn.executemany(
                    """
                    INSERT INTO relationships (
                        id, source_entity_id, source_entity_name, relation_type, 
                        target_entity_id, target_entity_name, chunk_id, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    [
                        (
                            r.id,
                            r.source_entity_id,
                            r.source_entity_name,
                            r.relation_type,
                            r.target_entity_id,
                            r.target_entity_name,
                            r.chunk_id,
                            r.confidence,
                        )
                        for r in relationships
                    ],
                )
            conn.commit()

        logger.info(
            "graph_elements_saved",
            num_entities=len(entities),
            num_relationships=len(relationships),
        )

    def search_entities(self, query: str, session_id: str) -> list[dict[str, Any]]:
        """Search entities by name using FTS5 match, filtered by session_id.

        Args:
            query: The keyword search term.
            session_id: Active session filter constraint.

        Returns:
            List of matching Entity records.
        """
        # Clean query for FTS5 (escape special characters)
        clean_query = query.replace('"', '').replace("'", "").strip()
        if not clean_query:
            return []

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT e.* FROM entities e
                JOIN entities_fts f ON f.rowid = e.rowid
                WHERE entities_fts MATCH ? AND e.id IN (
                    SELECT id FROM entities WHERE document_id IN (
                        SELECT id FROM documents WHERE session_id = ?
                    )
                )
                """,
                (f"{clean_query}*", session_id),
            ).fetchall()

            return [dict(r) for r in rows]

    def get_neighbors_by_entity_id(self, entity_id: str) -> list[dict[str, Any]]:
        """Find adjacent entities connected by one-hop relations."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT r.*, e1.name as source_name, e2.name as target_name 
                FROM relationships r
                JOIN entities e1 ON r.source_entity_id = e1.id
                JOIN entities e2 ON r.target_entity_id = e2.id
                WHERE r.source_entity_id = ? OR r.target_entity_id = ?
                """,
                (entity_id, entity_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_relationships_for_chunk(self, chunk_id: str) -> list[dict[str, Any]]:
        """Retrieve all graph edges associated with a specific text chunk."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM relationships WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchall()
            return [dict(r) for r in rows]
