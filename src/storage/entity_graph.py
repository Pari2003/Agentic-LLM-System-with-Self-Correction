"""
Neo4j Entity Graph Store (GraphRAG Database).

Responsible for storing the knowledge graph extracted from chunks:
1. Entity nodes (name, type, source chunk/doc, session_id)
2. Directed relationships/edges (source, target, relation label, confidence, session_id)

Design choices:
- Store in Neo4j to leverage Cypher queries and native graph search.
- Use session_id tag on all nodes and relationships to keep data session-isolated.
- Use a sanitization function to safely translate LLM relation names into Cypher relationship types.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

import structlog
from neo4j import GraphDatabase

from src.config import settings
from src.models.schemas import Entity, Relationship

logger = structlog.get_logger(__name__)


def sanitize_relationship_type(rel_type: str) -> str:
    """Sanitize relationship type name to be valid for Cypher labels."""
    # Allow only alphanumeric characters and underscores, uppercase
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", rel_type).strip("_").upper()
    return sanitized if sanitized else "RELATED_TO"


class EntityGraph:
    """Manages knowledge graph nodes and edges stored in Neo4j."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        db_path: Optional[Path] = None,
    ):
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.db_path = db_path or settings.sqlite_file

        logger.info("neo4j_driver_init_start", uri=self.uri, user=self.user)
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._init_constraints()

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()

    def __del__(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass

    def _init_constraints(self) -> None:
        """Create uniqueness constraints and indexes in Neo4j."""
        logger.info("neo4j_init_constraints_start")
        try:
            with self.driver.session() as session:
                # Uniqueness constraint on Entity ID
                session.run(
                    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
                )
                # Index on Entity name for fast searching/lookup
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                # Index on Entity session_id for fast cleanup and session-based lookups
                session.run(
                    "CREATE INDEX entity_session IF NOT EXISTS FOR (e:Entity) ON (e.session_id)"
                )
            logger.info("neo4j_init_constraints_complete")
        except Exception as e:
            logger.error("neo4j_init_constraints_failed", error=str(e))

    def _get_session_id_for_document(self, document_id: str) -> str:
        """Retrieve session_id for a given document_id from SQLite metadata store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT session_id FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                if row:
                    return row["session_id"]
        except Exception as e:
            logger.error("sqlite_session_lookup_failed", document_id=document_id, error=str(e))
        return "default_session"

    def _get_session_id_for_chunk(self, chunk_id: str) -> str:
        """Retrieve session_id for a given chunk_id from SQLite child metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT session_id FROM child_chunk_metadata WHERE id = ?", (chunk_id,)
                ).fetchone()
                if row:
                    return row["session_id"]
        except Exception as e:
            logger.error("sqlite_session_lookup_chunk_failed", chunk_id=chunk_id, error=str(e))
        return "default_session"

    def save_graph_elements(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Bulk save entities and relationships into the Neo4j graph store."""
        if not entities and not relationships:
            return

        # 1. Resolve sessions for entities
        entity_data = []
        for e in entities:
            session_id = self._get_session_id_for_document(e.document_id)
            entity_data.append(
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.entity_type.value,
                    "document_id": e.document_id,
                    "chunk_id": e.chunk_id,
                    "session_id": session_id,
                }
            )

        # 2. Save Entities in a transaction
        try:
            with self.driver.session() as session:
                entity_query = """
                UNWIND $entities AS ent
                MERGE (e:Entity {name: ent.name, session_id: ent.session_id})
                ON CREATE SET
                    e.id = ent.id,
                    e.type = ent.type,
                    e.document_id = ent.document_id,
                    e.chunk_id = ent.chunk_id
                """
                session.run(entity_query, entities=entity_data)

                # 3. Save Relationships
                for r in relationships:
                    session_id = self._get_session_id_for_chunk(r.chunk_id)
                    rel_label = sanitize_relationship_type(r.relation_type)

                    # Match by name and session_id to connect the correct nodes within the same session
                    rel_query = f"""
                    MATCH (s:Entity {{name: $source_name, session_id: $session_id}})
                    MATCH (t:Entity {{name: $target_name, session_id: $session_id}})
                    MERGE (s)-[r:{rel_label}]->(t)
                    SET r.id = $id,
                        r.chunk_id = $chunk_id,
                        r.session_id = $session_id,
                        r.confidence = $confidence
                    """
                    session.run(
                        rel_query,
                        source_name=r.source_entity_name,
                        target_name=r.target_entity_name,
                        session_id=session_id,
                        id=r.id,
                        chunk_id=r.chunk_id,
                        confidence=r.confidence,
                    )

            logger.info(
                "graph_elements_saved",
                num_entities=len(entities),
                num_relationships=len(relationships),
            )
        except Exception as e:
            logger.error("graph_elements_save_failed", error=str(e))
            raise

    def search_entities(self, query: str, session_id: str) -> list[dict[str, Any]]:
        """Search entities by name (substring match) filtered by session_id."""
        clean_query = query.strip()
        if not clean_query:
            return []

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {session_id: $session_id})
                    WHERE toLower(e.name) CONTAINS toLower($q)
                    RETURN e.id AS id, e.name AS name, e.type AS entity_type, e.document_id AS document_id, e.chunk_id AS chunk_id
                    """,
                    q=clean_query,
                    session_id=session_id,
                )
                return [record.data() for record in result]
        except Exception as e:
            logger.error("search_entities_failed", query=query, session_id=session_id, error=str(e))
            return []

    def get_neighbors_by_entity_id(self, entity_id: str) -> list[dict[str, Any]]:
        """Find adjacent entities connected by one-hop relations."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Entity)-[r]->(t:Entity)
                    WHERE s.id = $entity_id OR t.id = $entity_id
                    RETURN r.id AS id,
                           s.id AS source_entity_id,
                           s.name AS source_entity_name,
                           type(r) AS relation_type,
                           t.id AS target_entity_id,
                           t.name AS target_entity_name,
                           r.chunk_id AS chunk_id,
                           r.confidence AS confidence,
                           s.name AS source_name,
                           t.name AS target_name
                    """,
                    entity_id=entity_id,
                )
                return [record.data() for record in result]
        except Exception as e:
            logger.error("get_neighbors_failed", entity_id=entity_id, error=str(e))
            return []

    def get_relationships_for_chunk(self, chunk_id: str) -> list[dict[str, Any]]:
        """Retrieve all graph edges associated with a specific text chunk."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Entity)-[r]->(t:Entity)
                    WHERE r.chunk_id = $chunk_id
                    RETURN r.id AS id,
                           s.id AS source_entity_id,
                           s.name AS source_entity_name,
                           type(r) AS relation_type,
                           t.id AS target_entity_id,
                           t.name AS target_entity_name,
                           r.chunk_id AS chunk_id,
                           r.confidence AS confidence
                    """,
                    chunk_id=chunk_id,
                )
                return [record.data() for record in result]
        except Exception as e:
            logger.error("get_relationships_for_chunk_failed", chunk_id=chunk_id, error=str(e))
            return []
    
    def save_query_chunk_relationships(
            self,
        question: str,
        question_id: str,
        session_id: str,
        retrieved_chunks: list,
    ) -> None:
        """
        Store Question -> Chunk retrieval graph.
        """

        try:
            with self.driver.session() as session:

                # Create Question node
                session.run(
                    """
                    MERGE (q:Question {id:$qid})
                    SET q.text=$question,
                    q.session_id=$session_id
                    """,
                    qid=question_id,
                    question=question,
                    session_id=session_id,
                )

                # Create chunk nodes + relationships
                for rank, chunk in enumerate(retrieved_chunks, start=1):

                    session.run(
                        """
                        MERGE (c:Chunk {id:$chunk_id})
                        SET c.text=$chunk_text,
                        c.session_id=$session_id

                        WITH c

                        MATCH (q:Question {id:$qid})

                        MERGE (q)-[r:RETRIEVED]->(c)

                        SET r.score=$score,
                        r.rank=$rank
                        """,
                        qid=question_id,
                        chunk_id=chunk.chunk_id,
                        chunk_text=chunk.text[:1000],
                        session_id=session_id,
                        score=float(chunk.score),
                        rank=rank,
                    )

            logger.info(
                "query_chunk_graph_saved",
                question_id=question_id,
                num_chunks=len(retrieved_chunks),
            )

        except Exception as e:
            logger.error(
                "query_chunk_graph_failed",
                error=str(e),
            )
    
    def save_final_context_graph(
        self,
        question: str,
        question_id: str,
        session_id: str,
        final_contexts: list,
    ):
        """
        Save Question -> Final Parent Chunk graph
        """

        try:
            with self.driver.session() as session:

                # Create Question node
                session.run(
                    """
                    MERGE (q:Question {id:$qid})
                    SET q.text=$question,
                    q.session_id=$session_id
                    """,
                    qid=question_id,
                    question=question,
                    session_id=session_id,
                )

                for ctx in final_contexts:

                    session.run(
                        """
                        MERGE (p:ParentChunk {id:$pid})
                        SET p.text=$text,
                        p.session_id=$session_id

                        WITH p

                        MATCH (q:Question {id:$qid})

                        MERGE (q)-[r:USED_CONTEXT]->(p)

                        SET r.rank=$rank,
                        r.llm_score=$llm_score,
                        r.rrf_score=$rrf_score
                        """,

                        qid=question_id,

                        pid=ctx.parent_chunk.id,

                        text=ctx.parent_chunk.text[:1000],

                        session_id=session_id,

                        rank=ctx.final_rank,

                        llm_score=float(ctx.llm_relevance_score),

                        rrf_score=float(ctx.rrf_score),
                    )

            logger.info(
                "final_context_graph_saved",
                question_id=question_id,
                num_contexts=len(final_contexts),
            )

        except Exception as e:
            logger.error(
                "final_context_graph_failed",
                error=str(e),
            )

    def delete_session_graph(self, session_id: str) -> None:
        """Delete all entities and relationships associated with a session."""
        logger.info("neo4j_session_cleanup_start", session_id=session_id)
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (n {session_id: $session_id})
                    DETACH DELETE n
                    """,
                    session_id=session_id,
                )
            logger.info("neo4j_session_cleanup_complete", session_id=session_id)
        except Exception as e:
            logger.error("neo4j_session_cleanup_failed", session_id=session_id, error=str(e))
