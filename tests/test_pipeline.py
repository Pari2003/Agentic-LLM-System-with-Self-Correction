import asyncio
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agents.critic import Critic
from src.agents.generator import Generator
from src.agents.query_analyzer import QueryAnalyzer
from src.agents.refiner import Refiner
from src.agents.retriever import Retriever
from src.models.llm_client import OllamaClient
from src.models.schemas import (
    Chunk,
    ChunkLevel,
    ChunkMetadata,
    ChunkType,
    Citation,
    Document,
    Entity,
    EntityType,
    ParentChunk,
    QueryRequest,
    Relationship,
    Session,
)
from src.pipeline.orchestrator import Orchestrator
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
    print("=== STARTING PHASE 5 PIPELINE INTEGRATION TEST ===")

    # 1. Setup paths
    test_dir = Path("data/test_pipeline_run")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_db_path = test_dir / "test_metadata.db"
    test_chroma_path = test_dir / "test_chromadb"

    clean_test_paths(test_db_path, test_chroma_path)

    session_id = "test_pipeline_sess"
    doc_id = "test_pipeline_doc"

    try:
        temp_graph = EntityGraph(db_path=test_db_path)
        temp_graph.delete_session_graph(session_id)
        temp_graph.close()
    except Exception:
        pass

    try:
        # Initialize components
        print("\n[1] Initializing databases and clients...")
        llm_client = OllamaClient()
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
            filename="self_correction_study.pdf",
            title="Self-Correction in Retrieval-Augmented Generation",
            authors=["John Doe"],
            abstract="Active selection of context coupled with critic-based self-correction.",
            summary="Study on self-correction feedback loops.",
            total_pages=1,
            total_chunks=2,
            total_parent_chunks=1,
            total_child_chunks=2,
            session_id=session_id,
            ingested_at=now,
        )
        doc_store.save_document(test_doc)

        # Save parent chunk
        parent_id = "parent_1"
        test_parent = ParentChunk(
            id=parent_id,
            text=(
                "Agentic retrieval-augmented generation (RAG) represents a significant shift "
                "from passive context injection to active context selection. In this work, we "
                "propose a multi-stage retrieval pipeline coupled with self-correction feedback loops. "
                "The critic evaluates generated claims for factual overlap, and a refiner rewrites "
                "unverified statements. Our evaluation shows a 25 percent improvement in response reliability."
            ),
            document_id=doc_id,
            session_id=session_id,
            section_title="Abstract",
            page_numbers=[1],
            child_ids=["child_1", "child_2"],
            token_count=60,
        )
        doc_store.save_parent_chunks([test_parent])

        # Save child metadata
        child_ids = ["child_1", "child_2"]
        doc_store.save_child_metadata_batch(
            child_ids=child_ids,
            parent_ids=[parent_id, parent_id],
            doc_id=doc_id,
            session_id=session_id,
            types=["text", "text"],
            pages=[1, 1],
            sections=["Abstract", "Abstract"],
            indexes=[0, 1],
        )

        # Embed child chunks
        texts = [
            "Agentic retrieval-augmented generation (RAG) represents a shift to active context selection with self-correction.",
            "The critic evaluates claims for factual overlap, and a refiner rewrites statements to improve reliability by 25 percent.",
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
                        section_title="Abstract",
                        page_number=1,
                        parent_id=parent_id,
                        chunk_index=idx,
                    ),
                    embedding=emb,
                    token_count=18,
                )
            )
        vector_store.add_chunks(chunks)

        # Save entities
        entity_1 = Entity(
            id="ent_agentic_rag",
            name="Agentic RAG",
            entity_type=EntityType.MODEL,
            document_id=doc_id,
            chunk_id="child_1",
        )
        entity_2 = Entity(
            id="ent_self_correction",
            name="Self-Correction Loop",
            entity_type=EntityType.METHOD,
            document_id=doc_id,
            chunk_id="child_1",
        )
        rel = Relationship(
            id="rel_uses",
            source_entity_id=entity_1.id,
            source_entity_name=entity_1.name,
            relation_type="USES",
            target_entity_id=entity_2.id,
            target_entity_name=entity_2.name,
            chunk_id="child_1",
            confidence=0.95,
        )
        graph_store.save_graph_elements([entity_1, entity_2], [rel])
        print("  Database seeded successfully.")

        # Initialize agents and orchestrators
        query_analyzer = QueryAnalyzer(llm_client)
        hybrid_search = HybridSearch(vector_store, graph_store)
        parent_expander = ParentExpander(doc_store)
        llm_reranker = LLMReranker(llm_client)

        retriever = Retriever(
            llm_client=llm_client,
            query_analyzer=query_analyzer,
            hybrid_search=hybrid_search,
            parent_expander=parent_expander,
            llm_reranker=llm_reranker,
        )

        generator = Generator(llm_client, doc_store)
        critic = Critic(llm_client)
        refiner = Refiner(llm_client, doc_store)

        orchestrator = Orchestrator(
            retriever=retriever, generator=generator, critic=critic, refiner=refiner
        )

        # ─── Test Case 1: End-to-End Happy Path (No Hallucination) ───
        print("\n[2] Running Test Case 1: Grounded answer (No correction needed)...")
        request_happy = QueryRequest(
            question="Tell me about the reliability improvement in Agentic RAG.",
            session_id=session_id,
            max_sources=2,
            confidence_threshold=0.7,
            enable_self_correction=True,
            max_correction_iterations=2,
        )

        response_happy = await orchestrator.run(request_happy)
        print("\n=== Happy Path Answer ===")
        print(response_happy.answer)
        print("Citations:")
        for cit in response_happy.citations:
            print(
                f"  [{cit.source_id}] {cit.document_title} (Page {cit.page_number}): '{cit.relevant_text[:40]}...'"
            )
        print(f"Confidence Score: {response_happy.confidence_score}")
        print(f"Correction Iterations: {response_happy.correction_iterations}")

        assert response_happy.confidence_score >= 0.7
        assert response_happy.correction_iterations == 0
        assert len(response_happy.citations) > 0

        # ─── Test Case 2: Self-Correction Loop Triggered via Mocked Hallucination ───
        print("\n[3] Running Test Case 2: Hallucinated answer (Triggers self-correction)...")

        # Mock generator.generate to inject a hallucinated claim on first call
        original_generate = generator.generate
        call_count = 0

        async def mock_generate(query, contexts, session_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                print("  [Mock] Generating hallucinated response on first call...")
                # We inject an ungrounded claim that Albert Einstein invented Agentic RAG in 1905.
                hallucinated_text = (
                    "Agentic retrieval-augmented generation (RAG) represents a significant shift. "
                    "This system was invented by Albert Einstein in 1905 [1]."
                )
                citations = [
                    Citation(
                        source_id=1,
                        chunk_id="child_1",
                        document_id=doc_id,
                        document_title="Self-Correction in Retrieval-Augmented Generation",
                        section_title="Abstract",
                        page_number=1,
                        relevant_text="Agentic retrieval-augmented generation (RAG) represents a significant shift...",
                    )
                ]
                return hallucinated_text, citations
            else:
                print("  [Mock] Generating standard response on subsequent call...")
                return await original_generate(query, contexts, session_id)

        generator.generate = mock_generate

        request_correction = QueryRequest(
            question="Explain who designed and developed Agentic RAG.",
            session_id=session_id,
            max_sources=2,
            confidence_threshold=0.7,
            enable_self_correction=True,
            max_correction_iterations=2,
        )

        response_correction = await orchestrator.run(request_correction)

        print("\n=== Refined / Corrected Answer ===")
        print(response_correction.answer)
        print("Citations:")
        for cit in response_correction.citations:
            print(
                f"  [{cit.source_id}] {cit.document_title} (Page {cit.page_number}): '{cit.relevant_text[:40]}...'"
            )
        print(f"Confidence Score: {response_correction.confidence_score}")
        print(f"Correction Iterations: {response_correction.correction_iterations}")

        # Verify self-correction actually ran
        assert response_correction.correction_iterations > 0
        assert "Albert Einstein" not in response_correction.answer
        assert "1905" not in response_correction.answer
        assert response_correction.confidence_score >= 0.7

        # Teardown Neo4j graph data
        print("\nCleaning up session graph in Neo4j...")
        graph_store.delete_session_graph(session_id)

        # Close connections
        graph_store.close()
        await llm_client.close()

        print("\n=== PHASE 5 INTEGRATION TEST COMPLETED SUCCESSFULLY! ===")

    finally:
        print("\nCleaning up test directory...")
        clean_test_paths(test_db_path, test_chroma_path)
        shutil.rmtree(test_dir, ignore_errors=True)
        print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
