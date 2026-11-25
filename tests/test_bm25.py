
import pytest

from knowledge_engine.bm25 import BM25Retriever
from knowledge_engine.models import DocumentChunk


def make_chunk(
    chunk_id: str,
    content: str,
    index: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="test-doc",
        content=content,
        chunk_index=index,
        start_char=0,
        end_char=len(content),
    )


def test_bm25_ranks_relevant_chunk_first():
    chunks = [
        make_chunk(
            "chunk-a",
            "Python database indexing",
            0,
        ),
        make_chunk(
            "chunk-b",
            "Python programming basics",
            1,
        ),
        make_chunk(
            "chunk-c",
            "Gardening and soil",
            2,
        ),
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("python database", top_k=3)

    assert results[0][0].chunk_id == "chunk-a"
    assert results[0][1] > results[1][1]
    assert all(score > 0 for _, score in results)


def test_bm25_returns_no_results_for_unknown_terms():
    retriever = BM25Retriever([
        make_chunk("chunk-a", "Python programming", 0)
    ])

    assert retriever.search("unmatchedterm") == []


def test_bm25_rejects_invalid_parameters():
    chunk = make_chunk("chunk-a", "Python", 0)

    with pytest.raises(ValueError):
        BM25Retriever([chunk], k1=-1)

    with pytest.raises(ValueError):
        BM25Retriever([chunk], b=1.5)


def test_bm25_rejects_invalid_top_k():
    retriever = BM25Retriever([])

    with pytest.raises(ValueError):
        retriever.search("python", top_k=0)


def test_bm25_filters_by_category():
    chunks = [
        DocumentChunk(
            chunk_id="project-1",
            document_id="doc-1",
            content="Built a Python web application",
            chunk_index=0,
            start_char=0,
            end_char=31,
            metadata={"category": "projects"},
        ),
        DocumentChunk(
            chunk_id="skill-1",
            document_id="doc-2",
            content="Python programming and development",
            chunk_index=0,
            start_char=0,
            end_char=35,
            metadata={"category": "skills"},
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("Python", category="projects")

    assert len(results) == 1
    assert results[0][0].chunk_id == "project-1"


def test_bm25_without_category_searches_all_chunks():
    chunks = [
        DocumentChunk(
            chunk_id="project-1",
            document_id="doc-1",
            content="Built a Python web application",
            chunk_index=0,
            start_char=0,
            end_char=31,
            metadata={"category": "projects"},
        ),
        DocumentChunk(
            chunk_id="skill-1",
            document_id="doc-2",
            content="Python programming and development",
            chunk_index=0,
            start_char=0,
            end_char=35,
            metadata={"category": "skills"},
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("Python")

    assert len(results) == 2
