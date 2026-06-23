import asyncio
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Fix relative imports
sys.path.append(str(Path(__file__).parent))

from src.models.schemas import Session, Document, QueryRequest
from src.models.llm_client import OllamaClient
from src.storage.document_store import DocumentStore
from src.storage.vector_store import VectorStore
from src.storage.entity_graph import EntityGraph

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.parent_child_linker import ParentChildLinker
from src.ingestion.embedder import Embedder

from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.parent_expander import ParentExpander
from src.retrieval.llm_reranker import LLMReranker
from src.agents.query_analyzer import QueryAnalyzer
from src.agents.retriever import Retriever
from src.agents.generator import Generator
from src.agents.critic import Critic
from src.agents.refiner import Refiner
from src.pipeline.orchestrator import Orchestrator


async def ingest_document(file_path: str, session_id: str, doc_store, vector_store, llm_client) -> str:
    path = Path(file_path)
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".pdf":
        print(f"[-] Error: Invalid PDF file path: {file_path}")
        return None

    print(f"\n[*] Starting ingestion for: {path.name}")
    
    print("  [1/5] Parsing PDF...")
    from src.models.schemas import ChunkType
    parser = PDFParser()
    metadata = parser.extract_metadata(path)
    
    text_blocks = []
    for page_num, c_type, content, section_title in parser.parse(path):
        if c_type == ChunkType.TEXT:
            text_blocks.append(content)

    print("  [2/5] Semantic Chunking...")
    chunker = SemanticChunker()
    raw_chunks = []
    for text in text_blocks:
        groups = chunker.prepare_sentence_groups(text)
        if not groups:
            continue
        embs = await llm_client.embed(groups)
        chunks = chunker.chunk_text(text, embs, groups)
        raw_chunks.extend(chunks)

    print("  [3/5] Parent-Child Linking...")
    linker = ParentChildLinker()
    
    child_texts_with_meta = []
    for c in raw_chunks:
        child_texts_with_meta.append({
            "text": c,
            "page_number": 1,
            "section_title": "Document",
            "chunk_type": ChunkType.TEXT
        })

    child_chunks, parent_chunks = linker.link(
        document_id="pending",
        session_id=session_id,
        child_texts_with_meta=child_texts_with_meta
    )

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename=path.name,
        title=metadata.get("title"),
        authors=metadata.get("authors", []),
        abstract=metadata.get("abstract"),
        total_pages=metadata.get("total_pages", 0),
        total_chunks=len(parent_chunks) + len(child_chunks),
        total_parent_chunks=len(parent_chunks),
        total_child_chunks=len(child_chunks),
        total_tables=0,
        total_figures=0,
        session_id=session_id,
    )

    for pc in parent_chunks:
        pc.document_id = doc_id
    for cc in child_chunks:
        cc.metadata.document_id = doc_id

    print("  [4/5] Generating Embeddings...")
    embedder = Embedder(llm_client)
    child_chunks = await embedder.embed_chunks(child_chunks)

    print("  [5/5] Storing in Databases...")
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
        doc_id=doc_id,
        session_id=session_id,
        types=types,
        pages=pages,
        sections=sections,
        indexes=indexes,
    )

    vector_store.add_chunks(child_chunks)
    
    print(f"[+] Ingestion complete! Document ID: {doc_id}")
    return doc_id


async def main():
    print("=============================================")
    print("  Agentic RAG System - Terminal Interface  ")
    print("=============================================\n")

    print("[*] Initializing databases and AI models...")
    llm_client = OllamaClient()
    doc_store = DocumentStore()
    vector_store = VectorStore()
    graph_store = EntityGraph()
    
    # Initialize Pipeline Components
    query_analyzer = QueryAnalyzer(llm_client)
    hybrid_search = HybridSearch(vector_store, graph_store)
    parent_expander = ParentExpander(doc_store)
    llm_reranker = LLMReranker(llm_client)
    retriever = Retriever(llm_client, query_analyzer, hybrid_search, parent_expander, llm_reranker, graph_store)
    generator = Generator(llm_client, doc_store)
    critic = Critic(llm_client)
    refiner = Refiner(llm_client, doc_store)
    
    orchestrator = Orchestrator(retriever, generator, critic, refiner)

    # Create a fresh session
    existing_sessions = doc_store.get_existing_sessions()

    using_existing_session = False

    if existing_sessions:

        print("\nExisting sessions:\n")

        for i, sess in enumerate(existing_sessions, start=1):
            print(f"{i}. {sess['filename']} ({sess['id'][:8]})")

        print("0. Create New Session")
        choice = input("\nChoose: ").strip()

        if choice != "0":
            selected = existing_sessions[int(choice)-1]
            session_id = selected["id"]
            using_existing_session = True
            print(f"\n[+] Using previous session {session_id[:8]}")

        else:
            session_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(hours=24)
            session = Session(
                id=session_id,
                created_at=now,
                expires_at=expires_at,
                is_active=True,
            )
            doc_store.save_session(session)
            print(f"[+] New Session created ({session_id[:8]})")

    else:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=24)
        session = Session(
            id=session_id,
            created_at=now,
            expires_at=expires_at,
            is_active=True,
        )
        doc_store.save_session(session)
        print(f"[+] New Session created ({session_id[:8]})")

    # Ingestion phase
    while True:
        if using_existing_session:
            print("\n[*] Reusing previous session chunks.")
            pdf_path = "skip"
        else:
            pdf_path = input("Enter the path to a PDF file to analyze: ").strip()
        
        if pdf_path.lower() == 'skip':
            break
            
        # Clean path (strip quotes if pasted from explorer)
        pdf_path = pdf_path.strip('"').strip("'")
        
        doc_id = await ingest_document(pdf_path, session_id, doc_store, vector_store, llm_client)
        if doc_id:
            break

    print("\n=============================================")
    print("            Chat Session Started           ")
    print("=============================================")
    print("Type your questions below. Type 'exit' or 'quit' to close.\n")

    while True:
        question = input("\n[You]: ").strip()
        if not question:
            continue
        if question.lower() in ['exit', 'quit']:
            break
            
        print("\n[System]: Thinking... (This may take 30-60 seconds locally)")
        
        request = QueryRequest(
            question=question,
            session_id=session_id,
            max_sources=3,
            confidence_threshold=0.7,
            enable_self_correction=True,
            max_correction_iterations=2
        )
        
        try:
            response = await orchestrator.run(request)
            
            print("\n---------------------------------------------")
            print(f"[Agentic Answer]:\n{response.answer}")
            print("\n[Citations]:")
            if not response.citations:
                print("  No direct citations used.")
            for cit in response.citations:
                print(f"  [{cit.source_id}] Page {cit.page_number}: '{cit.relevant_text[:80]}...'")
            print(f"\n[Metrics]: Confidence: {response.confidence_score} | Self-Corrections: {response.correction_iterations}")
            print("---------------------------------------------")
            
        except Exception as e:
            print(f"\n[-] Error during query generation: {e}")

    print("\n[*] Cleaning up session...")
    graph_store.close()
    await llm_client.close()
    print("[+] Done. Goodbye!")

    

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[+] Exiting gracefully...")
