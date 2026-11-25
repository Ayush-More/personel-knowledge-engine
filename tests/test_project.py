
import pytest
from pydantic import ValidationError
import hashlib
import json
import numpy as np

from unittest.mock import Mock
from knowledge_engine.qdrant_retrieval import QdrantRetriever
from knowledge_engine.agent.runner import AgentRunner
from knowledge_engine.hybrid import HybridRetriever
from knowledge_engine.storage import save_chunk_jsonl
from knowledge_engine.config import Settings
from knowledge_engine.models import Document
from knowledge_engine.ingestion import ingest_text_file
from knowledge_engine.chunking import chunk_document , chunk_document_by_paragraph
from knowledge_engine.retrieval import KeywordRetriever, tokenize
from knowledge_engine.models import DocumentChunk
from knowledge_engine.vector_store import VectorStore
from knowledge_engine.pipeline import RetrievalPipeline
from knowledge_engine.models import RetrievedEvidence, ToolResult
from knowledge_engine.tool_registry import ToolRegistry
from knowledge_engine.agent.runner import AgentRunner

def test_settings_have_expected_defaults():
    settings = Settings()

    assert settings.app_name == "Personal Knowledge Engine"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50


def test_document_can_be_created():
    document = Document(
        document_id="doc-001",
        content="RAG combines retrieval with generation.",
        source="notes.txt",
    )

    assert document.document_id == "doc-001"
    assert document.metadata == {}
    assert document.created_at is not None


def test_document_rejects_empty_content():
    with pytest.raises(ValidationError):
        Document(
            document_id="doc-002",
            content="",
            source="empty.txt",
        )


def test_documents_do_not_share_metadata():
    first = Document(
        document_id="doc-001",
        content="First document",
        source="first.txt",
    )
    second = Document(
        document_id="doc-002",
        content="Second document",
        source="second.txt",
    )

    first.metadata["topic"] = "Python"

    assert second.metadata == {}


def test_ingest_text_file(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "This is my first knowledge document.",
        encoding="utf-8",
    )

    document = ingest_text_file(file_path)

    assert document.content == "This is my first knowledge document."
    assert document.source == str(file_path)
    assert document.metadata["file_type"] == "txt"
    assert document.document_id


def test_ingest_empty_file_fails_validation(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValidationError):
        ingest_text_file(file_path)


def test_chunk_document_preserves_content_and_order():
    document = Document(
        document_id="doc-001",
        content="abcdefghijklmnopqrstuvwxyz",
        source="test.txt",
    )

    chunks = chunk_document(
        document,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert [chunk.content for chunk in chunks] == [
        "abcdefghij",
        "ijklmnopqr",
        "qrstuvwxyz",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.document_id == "doc-001" for chunk in chunks)


def test_chunk_document_rejects_invalid_overlap():
    document = Document(
        document_id="doc-001",
        content="Some text",
        source="test.txt",
    )

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=10, chunk_overlap=10)


def test_short_document_produces_one_chunk():
    document = Document(
        document_id="doc-001",
        content="Short text",
        source="test.txt",
    )

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].content == "Short text"


def test_document_id_is_deterministic(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Stable document content.", encoding="utf-8")

    first = ingest_text_file(file_path)
    second = ingest_text_file(file_path)

    assert first.document_id == second.document_id


def test_paragraph_chunking_keeps_paragraphs_together():
    document = Document(
        document_id="doc-paragraph",
        content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        source="paragraphs.txt",
    )

    chunks = chunk_document_by_paragraph(document, chunk_size=40)

    assert [chunk.content for chunk in chunks] == [
        "First paragraph.\n\nSecond paragraph.",
        "Third paragraph.",
    ]


def test_paragraph_chunking_splits_oversized_paragraph():
    document = Document(
        document_id="doc-long",
        content="abcdefghij",
        source="long.txt",
    )

    chunks = chunk_document_by_paragraph(document, chunk_size=4)

    assert [chunk.content for chunk in chunks] == [
        "abcd",
        "efgh",
        "ij",
    ]


def test_paragraph_chunking_rejects_negative_size():
    document = Document(
        document_id="doc-invalid",
        content="A paragraph.",
        source="invalid.txt",
    )

    with pytest.raises(ValueError):
        chunk_document_by_paragraph(document, chunk_size=-1)


def test_save_chunk_jsonl(tmp_path):
    document = Document(
        document_id="doc-1",
        content="First paragraph.\n\nSecond paragraph.",
        source="test.txt",
    )

    chunks = chunk_document_by_paragraph(
        document,
        chunk_size=100,
    )

    output_path = tmp_path / "chunks.jsonl"
    save_chunk_jsonl(chunks, output_path)

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == len(chunks)

    first_record = json.loads(lines[0])
    assert first_record["content"] == chunks[0].content
    assert first_record["document_id"] == "doc-1"
    assert first_record["content_hash"] == hashlib.sha256(
        chunks[0].content.encode("utf-8")
    ).hexdigest()


def test_fixed_chunks_have_correct_offsets():
    text = "abcdefghij"
    document = Document(
        document_id="doc-offsets",
        content=text,
        source="test.txt",
    )

    chunks = chunk_document(
        document,
        chunk_size=4,
        chunk_overlap=0,
    )

    assert [chunk.content for chunk in chunks] == [
        "abcd",
        "efgh",
        "ij",
    ]

    for chunk in chunks:
        assert text[chunk.start_char:chunk.end_char] == chunk.content


def test_fixed_chunk_ids_are_deterministic():
    document = Document(
        document_id="doc-stable",
        content="abcdefghij",
        source="test.txt",
    )

    first_run = chunk_document(
        document,
        chunk_size=4,
        chunk_overlap=0,
    )
    second_run = chunk_document(
        document,
        chunk_size=4,
        chunk_overlap=0,
    )

    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]


def test_tokenize_normalizes_case_and_punctuation():
    assert tokenize("Python, DATABASE! python") == [
        "python",
        "database",
        "python",
    ]


def test_keyword_retriever_ranks_matching_chunks():
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Python database indexing",
            chunk_index=0,
            start_char=0,
            end_char=25,
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="Python programming basics",
            chunk_index=1,
            start_char=26,
            end_char=51,
        ),
        DocumentChunk(
            chunk_id="chunk-3",
            document_id="doc-1",
            content="Gardening and soil",
            chunk_index=2,
            start_char=52,
            end_char=70,
        ),
    ]

    retriever = KeywordRetriever(chunks)
    results = retriever.search("python database", top_k=3)

    assert results[0][0].chunk_id == "chunk-1"
    assert results[0][1] == 2.0
    assert results[1][0].chunk_id == "chunk-2"
    assert results[1][1] == 1.0
    assert len(results) == 2


def test_keyword_search_returns_no_results_for_unknown_terms():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Python programming",
        chunk_index=0,
        start_char=0,
        end_char=18,
    )

    retriever = KeywordRetriever([chunk])

    assert retriever.search("unmatchedterm") == []


def test_keyword_search_rejects_invalid_top_k():
    retriever = KeywordRetriever([])

    try:
        retriever.search("python", top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for top_k=0")



def test_document_chunk_preserves_source_and_metadata():
    chunk = DocumentChunk(
        chunk_id="chunk-123",
        document_id="doc-123",
        content="Built a project using FastAPI.",
        chunk_index=0,
        start_char=0,
        end_char=31,
        source="data/project-campusconnect.md",
        metadata={
            "file_type": "md",
            "category": "projects",
        },
    )

    assert chunk.source == "data/project-campusconnect.md"
    assert chunk.metadata["category"] == "projects"
    assert chunk.metadata["file_type"] == "md"


def test_fixed_chunking_preserves_source_and_metadata():
    document = Document(
        document_id="doc-123",
        content="ABCDEFGHIJ",
        source="data/project-campusconnect.md",
        metadata={
            "category": "projects",
            "file_type": "md",
        },
    )

    chunks = chunk_document(
        document,
        chunk_size=5,
        chunk_overlap=0,
    )

    assert len(chunks) == 2

    for chunk in chunks:
        assert chunk.source == "data/project-campusconnect.md"
        assert chunk.metadata["category"] == "projects"
        assert chunk.metadata["file_type"] == "md"


def test_paragraph_chunking_preserves_source_and_metadata():
    document = Document(
        document_id="doc-123",
        content="First paragraph.\n\nSecond paragraph.",
        source="data/project-campusconnect.md",
        metadata={"category": "projects"},
    )

    chunks = chunk_document_by_paragraph(
        document,
        chunk_size=100,
    )

    assert len(chunks) == 1
    assert chunks[0].source == "data/project-campusconnect.md"
    assert chunks[0].metadata["category"] == "projects"


def test_oversized_paragraph_chunking_preserves_metadata():
    document = Document(
        document_id="doc-123",
        content="ABCDEFGHIJ",
        source="data/project-campusconnect.md",
        metadata={"category": "projects"},
    )

    chunks = chunk_document_by_paragraph(
        document,
        chunk_size=5,
    )

    assert len(chunks) == 2

    for chunk in chunks:
        assert chunk.source == "data/project-campusconnect.md"
        assert chunk.metadata["category"] == "projects"

from knowledge_engine.storage import save_chunk_jsonl
from knowledge_engine.retrieval import load_chunks


def test_chunk_metadata_survives_save_and_load(tmp_path):
    chunk = DocumentChunk(
        chunk_id="chunk-persistence",
        document_id="doc-persistence",
        content="Built a project using FastAPI.",
        chunk_index=0,
        start_char=0,
        end_char=31,
        source="data/project-campusconnect.md",
        metadata={
            "file_type": "md",
            "category": "projects",
        },
    )

    output_file = tmp_path / "chunks.jsonl"

    save_chunk_jsonl([chunk], output_file)
    loaded_chunks = load_chunks(output_file)

    assert len(loaded_chunks) == 1
    assert loaded_chunks[0].source == "data/project-campusconnect.md"
    assert loaded_chunks[0].metadata["category"] == "projects"
    assert loaded_chunks[0].metadata["file_type"] == "md"

from knowledge_engine.ingestion import get_document_category


def test_document_category_mapping():
    assert get_document_category("project-campusconnect.md") == "projects"
    assert get_document_category("work-experience-kotak-life.md") == "experience"
    assert get_document_category("skills.md") == "skills"
    assert get_document_category("education.md") == "education"
    assert get_document_category("other-activities.md") == "activities"
    assert get_document_category("contact-information.md") == "contact"
    assert get_document_category("unknown-file.md") == "other"

def test_ingest_files_combines_chunks(tmp_path):
    from ingest import ingest_files

    file1 = tmp_path / "project-one.md"
    file2 = tmp_path / "skills.md"

    file1.write_text("Project information.", encoding="utf-8")
    file2.write_text("Python and FastAPI skills.", encoding="utf-8")

    chunks = ingest_files(
        input_files=[file1, file2],
        chunk_size=500,
    )

    assert len(chunks) == 2

    sources = {chunk.source for chunk in chunks}

    assert str(file1) in sources
    assert str(file2) in sources

    categories = {chunk.metadata["category"] for chunk in chunks}

    assert categories == {"projects", "skills"}

def test_vector_store_preserves_source_and_metadata(tmp_path):
    from knowledge_engine.vector_store import VectorStore

    store = VectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="test_metadata",
        vector_size=3,
    )

    try:
        store.upsert(
            chunk_id="chunk-123",
            text="Built a project using FastAPI.",
            vector=[0.1, 0.2, 0.3],
            source="data/project-campusconnect.md",
            metadata={
                "category": "projects",
                "file_type": "md",
            },
        )

        results = store.search(
            query_vector=[0.1, 0.2, 0.3],
            limit=1,
        )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "chunk-123"
        assert results[0]["text"] == "Built a project using FastAPI."

    finally:
        store.close()


def test_vector_store_filters_by_category(tmp_path):
    store = VectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="category_test",
        vector_size=3,
    )

    try:
        store.upsert(
            chunk_id="project-1",
            text="Built a Python application",
            vector=[1.0, 0.0, 0.0],
            source="project.md",
            metadata={"category": "projects"},
        )

        store.upsert(
            chunk_id="skill-1",
            text="Python programming skills",
            vector=[0.9, 0.1, 0.0],
            source="skills.md",
            metadata={"category": "skills"},
        )

        results = store.search(
            query_vector=[1.0, 0.0, 0.0],
            limit=5,
            category="projects",
        )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "project-1"
        assert results[0]["metadata"]["category"] == "projects"

    finally:
        store.close()


def test_qdrant_retriever_filters_by_category(tmp_path):
    class FakeEmbeddingModel:
        def encode(self, texts):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

    chunks = [
        DocumentChunk(
            chunk_id="project-1",
            document_id="doc-1",
            content="Built a Python application",
            chunk_index=0,
            start_char=0,
            end_char=28,
            metadata={"category": "projects"},
        ),
        DocumentChunk(
            chunk_id="skill-1",
            document_id="doc-2",
            content="Python programming skills",
            chunk_index=0,
            start_char=0,
            end_char=26,
            metadata={"category": "skills"},
        ),
    ]

    store = VectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="retriever_category_test",
        vector_size=3,
    )

    try:
        retriever = QdrantRetriever(
            chunks=chunks,
            store=store,
            embedding_model=FakeEmbeddingModel(),
        )

        retriever.index_chunks()

        results = retriever.search(
            query="Python",
            top_k=5,
            category="projects",
        )

        assert len(results) == 1
        assert results[0][0].chunk_id == "project-1"
        assert results[0][0].metadata["category"] == "projects"

    finally:
        store.close()

def test_hybrid_retriever_filters_by_category():
    from unittest.mock import Mock

    project_chunk = DocumentChunk(
        chunk_id="project-1",
        document_id="doc-1",
        content="Built a Python application",
        chunk_index=0,
        start_char=0,
        end_char=28,
        metadata={"category": "projects"},
    )

    skill_chunk = DocumentChunk(
        chunk_id="skill-1",
        document_id="doc-2",
        content="Python programming skills",
        chunk_index=0,
        start_char=0,
        end_char=26,
        metadata={"category": "skills"},
    )

    bm25 = Mock()
    semantic = Mock()

    bm25.search.return_value = [(project_chunk, 2.5)]
    semantic.search.return_value = [(project_chunk, 0.9)]

    hybrid = HybridRetriever(bm25=bm25, semantic=semantic)

    results = hybrid.search(
        query="Python",
        top_k=5,
        category="projects",
    )

    assert len(results) == 1
    assert results[0][0].chunk_id == "project-1"

    bm25.search.assert_called_once_with(
        "Python", 10, category="projects"
    )
    semantic.search.assert_called_once_with(
        "Python", 10, category="projects"
    )

def test_retrieval_pipeline_passes_category_to_hybrid():
    from unittest.mock import Mock

    project_chunk = DocumentChunk(
        chunk_id="project-1",
        document_id="doc-1",
        content="Built a Python application",
        chunk_index=0,
        start_char=0,
        end_char=28,
        metadata={"category": "projects"},
    )

    hybrid = Mock()
    reranker = Mock()

    candidates = [(project_chunk, 0.8)]
    hybrid.search.return_value = candidates
    reranker.rerank.return_value = candidates

    pipeline = RetrievalPipeline(
        hybrid=hybrid,
        reranker=reranker,
        candidate_k=10,
    )

    results = pipeline.search(
        query="Python",
        top_k=5,
        category="projects",
    )

    assert results == candidates

    hybrid.search.assert_called_once_with(
        "Python",
        top_k=10,
        category="projects",
    )

    reranker.rerank.assert_called_once_with(
        "Python",
        candidates,
        top_k=5,
    )

def test_tool_result_preserves_evidence():
    result = ToolResult(
        tool_name="search_projects",
        query="Python projects",
        evidence=[
            RetrievedEvidence(
                document_id = "a3a6427b2919f0dbefda2b8a288eb6dd36c2f1579bb39d5310ad60c1ed0ffa98",
                chunk_index = 0,    
                start_char = 0,
                end_char= 496, 
                metadata=  {"file_type": "md", "category": "general"},
                chunk_id="project-123",
                content="Built a Python application",
                source="data/project-campusconnect.md",
                score=0.87,
            )
        ],
        result_count=1,
    )

    assert result.tool_name == "search_projects"
    assert result.result_count == 1
    assert result.evidence[0].chunk_id == "project-123"
    assert result.evidence[0].score == 0.87

def test_search_projects_returns_structured_evidence():
    from unittest.mock import Mock

    from knowledge_engine.tools import KnowledgeTools

    chunk = DocumentChunk(
        chunk_id="project-1",
        document_id="doc-1",
        content="Built a Python application",
        chunk_index=0,
        start_char=0,
        end_char=28,
        source="data/project-campusconnect.md",
        metadata={"category": "projects"},
    )

    pipeline = Mock()
    pipeline.search.return_value = [(chunk, 0.85)]

    tools = KnowledgeTools(pipeline)

    result = tools.search_category(
        query="Python projects",
        top_k=3,
        category="projects",
        tool_name="search_projects"
    )

    assert result.tool_name == "search_projects"
    assert result.result_count == 1
    assert result.evidence[0].chunk_id == "project-1"
    # assert result.evidence[0].category == "projects"
    assert result.evidence[0].score == 0.85

    pipeline.search.assert_called_once_with(
        query="Python projects",
        top_k=3,
        category="projects",
    )

def test_search_projects_rejects_empty_query():
    from unittest.mock import Mock
    import pytest
    from knowledge_engine.tools import KnowledgeTools

    tools = KnowledgeTools(Mock())

    with pytest.raises(ValueError, match="Query must not be empty"):
        tools.search_category(query="   ")


def test_search_projects_rejects_invalid_top_k():
    from unittest.mock import Mock
    import pytest
    from knowledge_engine.tools import KnowledgeTools

    tools = KnowledgeTools(Mock())

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        tools.search_category(query="Python", top_k=0)

def test_search_skills_returns_structured_evidence():
    from unittest.mock import Mock

    from knowledge_engine.tools import KnowledgeTools

    chunk = DocumentChunk(
        chunk_id="skill-1",
        document_id="doc-1",
        content="Built a Python application",
        chunk_index=0,
        start_char=0,
        end_char=28,
        source="data/project-campusconnect.md",
        metadata={"category": "skills"},
    )

    pipeline = Mock()
    pipeline.search.return_value = [(chunk, 0.85)]

    tools = KnowledgeTools(pipeline)

    result = tools.search_category(
        query="Python skills",
        top_k=3,\
        tool_name="search_skills",
        category="skills"
    )

    assert result.tool_name == "search_skills"
    assert result.result_count == 1
    assert result.evidence[0].chunk_id == "skill-1"
    # assert result.evidence[0].category == "skills"
    assert result.evidence[0].score == 0.85

    pipeline.search.assert_called_once_with(
        query="Python skills",
        top_k=3,
        category="skills",
    )

from knowledge_engine.agent.evidence import deduplicate_evidence
from knowledge_engine.models import RetrievedEvidence


def test_deduplicate_evidence_keeps_highest_score():
    first = RetrievedEvidence(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Python experience",
        chunk_index=0,
        start_char=0,
        end_char=18,
        source="experience.md",
        metadata={"category": "experience"},
        score=0.75,
    )

    second = RetrievedEvidence(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Python experience",
        chunk_index=0,
        start_char=0,
        end_char=18,
        source="experience.md",
        metadata={"category": "experience"},
        score=0.92,
    )

    result = deduplicate_evidence([first, second])

    assert len(result) == 1
    assert result[0].chunk_id == "chunk-1"
    assert result[0].score == 0.92


def test_deduplicate_evidence_keeps_distinct_chunks():
    first = RetrievedEvidence(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Python experience",
        chunk_index=0,
        start_char=0,
        end_char=18,
        score=0.75,
    )

    second = RetrievedEvidence(
        chunk_id="chunk-2",
        document_id="doc-1",
        content="FastAPI experience",
        chunk_index=1,
        start_char=18,
        end_char=36,
        score=0.85,
    )

    result = deduplicate_evidence([first, second])

    assert len(result) == 2

def test_deduplicate_evidence_handles_empty_input():
    result = deduplicate_evidence([])

    assert result == []

def test_deduplicate_evidence_preserves_metadata_from_best_result():
    first = RetrievedEvidence(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Python experience",
        chunk_index=0,
        start_char=0,
        end_char=18,
        source="old.md",
        metadata={"category": "general"},
        score=0.75,
    )

    second = RetrievedEvidence(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Python experience",
        chunk_index=0,
        start_char=0,
        end_char=18,
        source="experience.md",
        metadata={"category": "experience"},
        score=0.95,
    )

    result = deduplicate_evidence([first, second])

    assert result[0].score == 0.95
    assert result[0].source == "experience.md"
    assert result[0].metadata == {"category": "experience"}


def test_runner_initializes_with_dependencies():
    client = Mock()
    registry = ToolRegistry()

    runner = AgentRunner(client, registry)

    assert runner.client is client
    assert runner.registry is registry
    assert runner.max_tool_calls == 5


def test_runner_rejects_non_positive_tool_limit():
    with pytest.raises(ValueError):
        AgentRunner(Mock(), ToolRegistry(), max_tool_calls=0)


def test_build_prompt_includes_question_and_history():
    client = Mock()
    registry = ToolRegistry()

    runner = AgentRunner(client, registry)

    prompt = runner._build_prompt(
        question="What projects use Python?",
        history=["Called search_projects", "Found 3 evidence chunks"],
    )

    assert "What projects use Python?" in prompt
    assert "Called search_projects" in prompt
    assert "Found 3 evidence chunks" in prompt

def test_execute_tool_calls_registered_function():
    from unittest.mock import Mock

    from knowledge_engine.tool_registry import ToolDefinition

    client = Mock()
    registry = ToolRegistry()

    mock_function = Mock(return_value="tool result")

    registry.register(
        ToolDefinition(
            name="search_projects",
            description="Search projects",
            function=mock_function,
        )
    )

    runner = AgentRunner(client, registry)

    result = runner._execute_tool(
        "search_projects",
        {"query": "Python"},
    )

    assert result == "tool result"
    mock_function.assert_called_once_with(query="Python")