import asyncio
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agents.query_analyzer import QueryAnalyzer
from src.agents.retriever import Retriever
from src.models.llm_client import OllamaClient
from src.models.schemas import (
    Chunk,
    ChunkLevel,
    ChunkMetadata,
    ChunkType,
    Document,
    Entity,
    EntityType,
    ParentChunk,
    Relationship,
    Session,
)
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.llm_reranker import LLMReranker
from src.retrieval.parent_expander import ParentExpander
from src.storage.document_store import DocumentStore
from src.storage.entity_graph import EntityGraph
from src.storage.vector_store import VectorStore


def clean_test_paths(db_path: Path, chroma_path: Path):
    """Safely remove existing test database files and directories."""
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass
        for suffix in ["-wal", "-shm"]:
            wal_file = db_path.with_name(db_path.name + suffix)
            if wal_file.exists():
                try:
                    wal_file.unlink()
                except OSError:
                    pass
    if chroma_path.exists():
        shutil.rmtree(chroma_path, ignore_errors=True)


async def main():
    print("=== STARTING PHASE 4 RETRIEVAL PIPELINE TEST ===")

    # 1. Setup paths
    test_dir = Path("data/test_retrieval_run")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_db_path = test_dir / "test_metadata.db"
    test_chroma_path = test_dir / "test_chromadb"

    clean_test_paths(test_db_path, test_chroma_path)

    # Pre-clean Neo4j test session
    session_id = "test_retrieval_sess"
    doc_id = "test_doc_rag"

    try:
        temp_graph = EntityGraph(db_path=test_db_path)
        temp_graph.delete_session_graph(session_id)
        temp_graph.close()
    except Exception:
        pass

    try:
        # Initialize LLM Client
        print("\n[1] Initializing OllamaClient...")
        llm_client = OllamaClient()

        # Initialize databases
        print("Initializing database stores...")
        doc_store = DocumentStore(db_path=test_db_path)
        vector_store = VectorStore(persist_dir=str(test_chroma_path))
        graph_store = EntityGraph(db_path=test_db_path)

        # Save a test Session
        now = datetime.now(timezone.utc)
        test_session = Session(
            id=session_id,
            created_at=now,
            expires_at=now + timedelta(hours=2),
            document_count=1,
            query_count=0,
            is_active=True,
        )
        doc_store.save_session(test_session)

        # Save a test Document
        test_doc = Document(
            id=doc_id,
            filename="agent_paper.pdf",
            title="Agentic LLM Systems: A New Paradigm",
            authors=["Ashish Vaswani"],
            abstract="In this work, we propose a multi-stage retrieval pipeline.",
            summary="Intro to agentic RAG.",
            total_pages=2,
            total_chunks=3,
            total_parent_chunks=1,
            total_child_chunks=3,
            session_id=session_id,
            ingested_at=now,
        )
        doc_store.save_document(test_doc)

        # Save parent chunks
        parent_id_1 = "parent_1"
        test_parent_1 = ParentChunk(
            id=parent_id_1,
            text="Agentic retrieval-augmented generation (RAG) represents a significant shift from passive context injection to active context selection. In this work, we propose a multi-stage retrieval pipeline coupled with self-correction feedback loops. Our evaluation shows a 25 percent improvement in reliability.",
            document_id=doc_id,
            session_id=session_id,
            section_title="Abstract",
            page_numbers=[1],
            child_ids=["child_1", "child_2"],
            token_count=50,
        )
        parent_id_2 = "parent_2"
        test_parent_2 = ParentChunk(
            id=parent_id_2,
            text="The architecture consists of three components. First, the ingestion pipeline splits text into semantic units. Second, the hybrid retrieval module combines dense vectors with sparse BM25 scores. Third, a self-correction loop verifies all generated claims.",
            document_id=doc_id,
            session_id=session_id,
            section_title="Methodology",
            page_numbers=[2],
            child_ids=["child_3"],
            token_count=50,
        )
        doc_store.save_parent_chunks([test_parent_1, test_parent_2])

        # Save child metadata
        child_ids = ["child_1", "child_2", "child_3"]
        parent_ids = [parent_id_1, parent_id_1, parent_id_2]
        doc_store.save_child_metadata_batch(
            child_ids=child_ids,
            parent_ids=parent_ids,
            doc_id=doc_id,
            session_id=session_id,
            types=["text"] * 3,
            pages=[1, 1, 2],
            sections=["Abstract", "Abstract", "Methodology"],
            indexes=[0, 1, 0],
        )

        # Save child vector chunks with embeddings
        # We will embed real text snippets so search behaves realistically
        texts = [
            "Agentic retrieval-augmented generation (RAG) represents a significant shift from passive context injection.",
            "In this work, we propose a multi-stage retrieval pipeline coupled with self-correction feedback loops.",
            "The architecture consists of three components: ingestion pipeline, hybrid retrieval module, and self-correction loop.",
        ]
        embeddings = await llm_client.embed(texts)

        chunks = []
        for idx, (cid, text, emb) in enumerate(zip(child_ids, texts, embeddings)):
            chunks.append(
                Chunk(
                    id=cid,
                    text=text,
                    metadata=ChunkMetadata(
                        document_id=doc_id,
                        session_id=session_id,
                        chunk_type=ChunkType.TEXT,
                        chunk_level=ChunkLevel.CHILD,
                        section_title="Abstract" if idx < 2 else "Methodology",
                        page_number=1 if idx < 2 else 2,
                        parent_id=parent_ids[idx],
                        chunk_index=idx if idx < 2 else 0,
                    ),
                    embedding=emb,
                    token_count=15,
                )
            )
        vector_store.add_chunks(chunks)

        # Save entities in Graph
        entity_1 = Entity(
            id="ent_rag",
            name="RAG System",
            entity_type=EntityType.MODEL,
            document_id=doc_id,
            chunk_id="child_1",
        )
        entity_2 = Entity(
            id="ent_self_correction",
            name="Self-Correction Loop",
            entity_type=EntityType.METHOD,
            document_id=doc_id,
            chunk_id="child_2",
        )
        rel = Relationship(
            id="rel_includes",
            source_entity_id=entity_1.id,
            source_entity_name=entity_1.name,
            relation_type="INCLUDES",
            target_entity_id=entity_2.id,
            target_entity_name=entity_2.name,
            chunk_id="child_2",
            confidence=0.9,
        )
        graph_store.save_graph_elements([entity_1, entity_2], [rel])
        print("  Database seeded with documents, parents, child vectors, and graph elements.")

        # ─── 2. Test QueryAnalyzer ───
        print("\n[2] Testing QueryAnalyzer...")
        query_analyzer = QueryAnalyzer(llm_client)
        analysis = await query_analyzer.analyze("Tell me about RAG systems with self-correction.")
        print(f"  Complexity: {analysis.complexity}")
        print(f"  Extracted Entities: {analysis.extracted_entities}")
        print(f"  Search Queries: {analysis.search_queries}")
        print(f"  Intent: {analysis.intent}")

        assert isinstance(analysis.search_queries, list)
        assert len(analysis.search_queries) > 0

        # ─── 3. Test HybridSearch ───
        print("\n[3] Testing HybridSearch (Dense + Sparse + Graph Boost)...")
        hybrid_search = HybridSearch(vector_store, graph_store)

        query_str = "RAG systems with self-correction"
        query_vector = await llm_client.embed_single(query_str)

        child_results = await hybrid_search.search(
            query=query_str,
            query_vector=query_vector,
            session_id=session_id,
            extracted_entities=["RAG System", "Self-Correction Loop"],
            top_k=2,
        )

        print(f"  Found {len(child_results)} child chunks:")
        for idx, res in enumerate(child_results):
            print(
                f"    Rank {idx + 1}: ID={res.chunk_id} | Score={res.score} | Source={res.source} | Text='{res.text[:40]}...'"
            )

        assert len(child_results) > 0
        # Check that we retrieved correct chunk IDs
        assert any(r.chunk_id == "child_1" for r in child_results) or any(
            r.chunk_id == "child_2" for r in child_results
        )

        # ─── 4. Test ParentExpander ───
        print("\n[4] Testing ParentExpander (Child -> Parent expansion)...")
        parent_expander = ParentExpander(doc_store)
        parent_candidates = parent_expander.expand(child_results, top_n=2)

        print(f"  Expanded to {len(parent_candidates)} parent chunk candidates:")
        for idx, cand in enumerate(parent_candidates):
            print(
                f"    Candidate {idx + 1}: ID={cand.parent_chunk.id} | RRF Score={cand.rrf_score} | Text='{cand.parent_chunk.text[:50]}...'"
            )

        assert len(parent_candidates) > 0
        # Parent chunk should have correct ID mapping
        assert parent_candidates[0].parent_chunk.id in ["parent_1", "parent_2"]

        # ─── 5. Test LLMReranker ───
        print("\n[5] Testing LLMReranker...")
        llm_reranker = LLMReranker(llm_client)
        reranked_contexts = await llm_reranker.rerank(
            query=query_str, contexts=parent_candidates, top_k=2
        )

        print("  Reranked top contexts:")
        for idx, ctx in enumerate(reranked_contexts):
            print(
                f"    Rank {ctx.final_rank}: ID={ctx.parent_chunk.id} | LLM Score={ctx.llm_relevance_score} | Text='{ctx.parent_chunk.text[:50]}...'"
            )

        assert len(reranked_contexts) > 0
        assert reranked_contexts[0].final_rank == 1

        # ─── 6. Test Retriever Orchestrator ───
        print("\n[6] Testing Retriever Orchestrator...")
        retriever = Retriever(
            llm_client=llm_client,
            query_analyzer=query_analyzer,
            hybrid_search=hybrid_search,
            parent_expander=parent_expander,
            llm_reranker=llm_reranker,
        )

        contexts, query_analysis, latencies = await retriever.retrieve(
            query="Explain the self-correction loops in active context selection.",
            session_id=session_id,
            max_sources=2,
        )

        print(f"  Orchestrated Retrieval returned {len(contexts)} contexts:")
        for idx, ctx in enumerate(contexts):
            print(
                f"    Result {idx + 1}: ID={ctx.parent_chunk.id} | Score={ctx.llm_relevance_score} | Text='{ctx.parent_chunk.text[:50]}...'"
            )

        print("\n  Latencies:")
        for k, v in latencies.items():
            print(f"    {k}: {v:.2f} ms")

        assert len(contexts) > 0
        assert "query_analysis_ms" in latencies
        assert "query_embedding_ms" in latencies
        assert "hybrid_search_ms" in latencies
        assert "parent_expansion_ms" in latencies
        assert "llm_rerank_ms" in latencies
        assert "total_ms" in latencies

        # Clean up session graph in Neo4j
        print("\nCleaning up session graph in Neo4j...")
        graph_store.delete_session_graph(session_id)

        # Close connection
        graph_store.close()
        await llm_client.close()

        print("\n=== PHASE 4 RETRIEVAL PIPELINE FULLY VERIFIED! ===")

    finally:
        # Clean up database files
        print("\nCleaning up test directory...")
        clean_test_paths(test_db_path, test_chroma_path)
        shutil.rmtree(test_dir, ignore_errors=True)
        print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
