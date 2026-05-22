import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time

from src.models.schemas import (
    Session, Document, ParentChunk, Chunk, ChunkMetadata, ChunkType, ChunkLevel,
    Entity, Relationship, EntityType
)
from src.storage.document_store import DocumentStore
from src.storage.vector_store import VectorStore
from src.storage.entity_graph import EntityGraph
from src.storage.session_manager import SessionManager

def clean_test_paths(db_path: Path, chroma_path: Path):
    """Safely remove existing test database files and directories."""
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass
        # SQLite WAL mode files
        for suffix in ["-wal", "-shm"]:
            wal_file = db_path.with_name(db_path.name + suffix)
            if wal_file.exists():
                try:
                    wal_file.unlink()
                except OSError:
                    pass
    if chroma_path.exists():
        shutil.rmtree(chroma_path, ignore_errors=True)

def main():
    print("=== STARTING PHASE 3 STORAGE TEST ===")
    
    # 1. Setup paths
    test_dir = Path("data/test_storage_run")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_db_path = test_dir / "test_metadata.db"
    test_chroma_path = test_dir / "test_chromadb"
    
    clean_test_paths(test_db_path, test_chroma_path)
    
    # Pre-clean Neo4j test sessions
    try:
        temp_graph = EntityGraph(db_path=test_db_path)
        temp_graph.delete_session_graph("test_sess_1")
        temp_graph.delete_session_graph("expired_sess_999")
        temp_graph.close()
    except Exception:
        pass

    try:
        # 2. Test DocumentStore
        print("\n[1] Initializing DocumentStore...")
        doc_store = DocumentStore(db_path=test_db_path)
        
        print("Testing session creation and retrieval...")
        session_id = "test_sess_1"
        now = datetime.now(timezone.utc)
        test_session = Session(
            id=session_id,
            created_at=now,
            expires_at=now + timedelta(hours=2),
            document_count=0,
            query_count=0,
            is_active=True
        )
        doc_store.save_session(test_session)
        
        retrieved_sess = doc_store.get_session(session_id)
        assert retrieved_sess is not None, "Failed to retrieve session"
        assert retrieved_sess.id == session_id, "Retrieved incorrect session ID"
        print(f"  Session successfully saved and retrieved: {retrieved_sess.id}")

        print("Testing document metadata creation and retrieval...")
        doc_id = "test_doc_1"
        test_doc = Document(
            id=doc_id,
            filename="attention_is_all_you_need.pdf",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            abstract="We propose a new simple network architecture, the Transformer...",
            summary="A seminal paper introducing the self-attention mechanism.",
            total_pages=15,
            total_chunks=10,
            total_parent_chunks=2,
            total_child_chunks=8,
            total_tables=1,
            total_figures=1,
            session_id=session_id,
            ingested_at=now
        )
        doc_store.save_document(test_doc)
        
        retrieved_doc = doc_store.get_document(doc_id)
        assert retrieved_doc is not None, "Failed to retrieve document"
        assert retrieved_doc.title == "Attention Is All You Need", "Document title mismatch"
        assert "Ashish Vaswani" in retrieved_doc.authors, "Document authors mismatch"
        print(f"  Document successfully saved and retrieved: '{retrieved_doc.title}' by {retrieved_doc.authors}")

        print("Testing parent chunk insertion and retrieval...")
        parent_id = "test_parent_1"
        test_parent = ParentChunk(
            id=parent_id,
            text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            document_id=doc_id,
            session_id=session_id,
            section_title="Introduction",
            page_numbers=[1, 2],
            child_ids=["child_1", "child_2"],
            token_count=100
        )
        doc_store.save_parent_chunks([test_parent])
        
        retrieved_parent = doc_store.get_parent_chunk(parent_id)
        assert retrieved_parent is not None, "Failed to retrieve parent chunk"
        assert retrieved_parent.section_title == "Introduction", "Parent chunk section title mismatch"
        print(f"  Parent Chunk successfully saved and retrieved: '{retrieved_parent.text[:40]}...'")

        print("Testing child chunk metadata registration and linking...")
        child_ids = ["child_1", "child_2"]
        parent_ids = [parent_id, parent_id]
        types = [ChunkType.TEXT, ChunkType.TEXT]
        pages = [1, 2]
        sections = ["Introduction", "Introduction"]
        indexes = [0, 1]
        
        doc_store.save_child_metadata_batch(
            child_ids=child_ids,
            parent_ids=parent_ids,
            doc_id=doc_id,
            session_id=session_id,
            types=[t.value for t in types],
            pages=pages,
            sections=sections,
            indexes=indexes
        )
        
        linked_parent = doc_store.get_parent_by_child_id("child_1")
        assert linked_parent is not None, "Failed to resolve parent via child_1"
        assert linked_parent.id == parent_id, "Linked parent ID mismatch"
        assert "child_1" in linked_parent.child_ids, "Child ID missing from parent's children lists"
        print(f"  Child-to-parent mapping resolved successfully: child_1 -> {linked_parent.id}")

        # 3. Test EntityGraph
        print("\n[2] Initializing EntityGraph...")
        graph_store = EntityGraph(db_path=test_db_path)
        
        print("Testing entity and relationship insertions...")
        entity_1 = Entity(
            id="ent_transformer",
            name="Transformer Model",
            entity_type=EntityType.MODEL,
            document_id=doc_id,
            chunk_id="child_1"
        )
        entity_2 = Entity(
            id="ent_self_attention",
            name="Self-Attention",
            entity_type=EntityType.METHOD,
            document_id=doc_id,
            chunk_id="child_1"
        )
        rel = Relationship(
            id="rel_uses",
            source_entity_id=entity_1.id,
            source_entity_name=entity_1.name,
            relation_type="USES",
            target_entity_id=entity_2.id,
            target_entity_name=entity_2.name,
            chunk_id="child_1",
            confidence=0.95
        )
        
        graph_store.save_graph_elements(entities=[entity_1, entity_2], relationships=[rel])
        print("  Graph elements saved.")

        print("Testing FTS5 search on entities...")
        search_results = graph_store.search_entities("Transformer", session_id)
        assert len(search_results) > 0, "No entities matched FTS search"
        assert search_results[0]["name"] == "Transformer Model", "FTS search matched wrong entity"
        print(f"  FTS Search matches: {search_results}")

        print("Testing one-hop neighborhood retrieval...")
        neighbors = graph_store.get_neighbors_by_entity_id("ent_transformer")
        assert len(neighbors) > 0, "No neighbors found for ent_transformer"
        assert neighbors[0]["relation_type"] == "USES", "Incorrect relationship returned"
        print(f"  One-hop neighbors: {neighbors}")

        print("Testing relationships for chunk...")
        chunk_rels = graph_store.get_relationships_for_chunk("child_1")
        assert len(chunk_rels) > 0, "No relationships found for child_1"
        print(f"  Chunk relations: {chunk_rels}")

        # 4. Test VectorStore
        print("\n[3] Initializing VectorStore...")
        vector_store = VectorStore(persist_dir=str(test_chroma_path))
        
        print("Testing adding child chunks to VectorStore...")
        dummy_embedding = [0.1] * 768
        child_chunk_1 = Chunk(
            id="child_1",
            text="The Transformer uses self-attention mechanisms.",
            metadata=ChunkMetadata(
                document_id=doc_id,
                session_id=session_id,
                chunk_type=ChunkType.TEXT,
                chunk_level=ChunkLevel.CHILD,
                section_title="Introduction",
                page_number=1,
                parent_id=parent_id,
                chunk_index=0
            ),
            embedding=dummy_embedding,
            token_count=6
        )
        child_chunk_2 = Chunk(
            id="child_2",
            text="Recurrent networks process tokens sequentially.",
            metadata=ChunkMetadata(
                document_id=doc_id,
                session_id=session_id,
                chunk_type=ChunkType.TEXT,
                chunk_level=ChunkLevel.CHILD,
                section_title="Introduction",
                page_number=2,
                parent_id=parent_id,
                chunk_index=1
            ),
            embedding=[0.2] * 768,
            token_count=6
        )
        vector_store.add_chunks([child_chunk_1, child_chunk_2])
        print("  Chunks added to VectorStore.")

        print("Testing querying similarity with session partition filter...")
        # Query with embedding close to dummy_embedding
        query_results = vector_store.query_similarity(
            query_vector=[0.11] * 768,
            session_id=session_id,
            top_k=2
        )
        assert len(query_results) > 0, "No results returned from ChromaDB"
        assert query_results[0].chunk_id == "child_1", f"Incorrect top result: {query_results[0].chunk_id}"
        # Score calculation is 1 - distance. Distances are small, so score should be close to 1.0.
        print(f"  Query result: Chunk ID={query_results[0].chunk_id}, Score={query_results[0].score:.4f}, Text='{query_results[0].text}'")

        print("Testing isolation from other sessions...")
        other_query_results = vector_store.query_similarity(
            query_vector=[0.11] * 768,
            session_id="different_session_id",
            top_k=2
        )
        assert len(other_query_results) == 0, f"Query returned chunks from another session: {other_query_results}"
        print("  Session isolation verified (0 results for different_session_id).")

        # 5. Test SessionManager
        print("\n[4] Initializing SessionManager...")
        # Use a short interval for testing background cleanups
        sess_mgr = SessionManager(
            doc_store=doc_store,
            vector_store=vector_store,
            graph_store=graph_store,
            ttl_hours=1,
            cleanup_interval_mins=1,
            start_scheduler=True
        )
        
        print("Testing session creation via SessionManager...")
        managed_sess = sess_mgr.create_session()
        assert managed_sess is not None, "Failed to create session via SessionManager"
        print(f"  Created session: {managed_sess.id}, expires at {managed_sess.expires_at}")
        
        print("Testing validation and TTL extension (sliding window)...")
        initial_expiry = managed_sess.expires_at
        time.sleep(0.1) # tiny pause
        is_valid = sess_mgr.validate_session(managed_sess.id)
        assert is_valid, "Validation failed for newly created session"
        
        updated_sess = doc_store.get_session(managed_sess.id)
        assert updated_sess.expires_at > initial_expiry, "TTL did not extend on validation"
        print(f"  Validation success. New expiry: {updated_sess.expires_at}")

        print("Testing session expiration behavior...")
        # Inject an expired session manually into the database
        expired_session_id = "expired_sess_999"
        expired_session = Session(
            id=expired_session_id,
            created_at=now - timedelta(hours=5),
            expires_at=now - timedelta(hours=3),
            document_count=0,
            is_active=True
        )
        doc_store.save_session(expired_session)
        
        # Add vector chunks for the expired session
        expired_chunk = Chunk(
            id="expired_child",
            text="Expired session data.",
            metadata=ChunkMetadata(
                document_id=doc_id,
                session_id=expired_session_id,
                chunk_type=ChunkType.TEXT,
                chunk_level=ChunkLevel.CHILD,
                page_number=1,
                chunk_index=0
            ),
            embedding=[0.5] * 768,
            token_count=3
        )
        vector_store.add_chunks([expired_chunk])
        
        # Verify validation fails on expired session
        is_expired_valid = sess_mgr.validate_session(expired_session_id)
        assert not is_expired_valid, "Expired session was incorrectly validated as active"
        print("  Validation correctly failed for expired session.")
        
        # Verify it got cleaned up
        assert doc_store.get_session(expired_session_id) is None, "Expired session SQLite record was not purged"
        expired_vect_query = vector_store.query_similarity([0.5] * 768, expired_session_id, top_k=1)
        assert len(expired_vect_query) == 0, "Expired session vector chunks were not purged"
        print("  Auto-purge of expired session verified.")

        print("Testing manual session closing (explicit cleanup)...")
        # Verify active session has records first
        assert doc_store.get_session(session_id) is not None, "Original session not found before close"
        sess_mgr.close_session(session_id)
        
        # Verify SQLite cascade deletes
        assert doc_store.get_session(session_id) is None, "Session record still exists after close"
        assert doc_store.get_document(doc_id) is None, "Document record was not cascadingly deleted"
        assert doc_store.get_parent_chunk(parent_id) is None, "Parent chunk was not cascadingly deleted"
        
        # Verify VectorStore cleanup
        orig_vect_query = vector_store.query_similarity(dummy_embedding, session_id, top_k=2)
        assert len(orig_vect_query) == 0, "Vector chunks still exist in ChromaDB after close"
        
        # Verify EntityGraph cleanup in Neo4j
        post_close_entities = graph_store.search_entities("Transformer", session_id)
        assert len(post_close_entities) == 0, "Entities still exist in Neo4j after close"
        
        print("  Manual session close and cascading deletes verified (SQLite, ChromaDB, and Neo4j).")
        
        print("\nShutting down SessionManager scheduler...")
        sess_mgr.shutdown()
        
        print("\n=== PHASE 3 STORAGE INTEGRATION FULLY VERIFIED! ===")

    finally:
        # Close graph_store connection
        try:
            graph_store.close()
        except Exception:
            pass
        # 6. Cleanup files
        print("\nCleaning up test directory...")
        clean_test_paths(test_db_path, test_chroma_path)
        shutil.rmtree(test_dir, ignore_errors=True)
        print("Cleanup done.")

if __name__ == "__main__":
    main()
