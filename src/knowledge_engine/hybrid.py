from collections import defaultdict

from knowledge_engine.bm25 import BM25Retriever
from knowledge_engine.models import DocumentChunk
from knowledge_engine.qdrant_retrieval import QdrantRetriever

class HybridRetriever:
    """Combine BM25 and semantic rankings using RRF."""
    def __init__(self , bm25: BM25Retriever , semantic: QdrantRetriever , rrf_k:int = 60):

        if rrf_k <= 0:
            raise ValueError("rrf_k value must be greater then zero")

        self.bm25 = bm25
        self.semantic = semantic
        self.rrf_k = rrf_k

    def search(self , query: str , top_k: int = 5 , category: str | None = None) -> list[tuple[DocumentChunk , float]]:

        if top_k <= 0:
            raise ValueError("top_k must be greater then zero")

        if not query.strip():
            return []

        candidate_k = max(top_k * 2 , 10)

        bm25_results = self.bm25.search(query , candidate_k , category=category)
        semantic_results = self.semantic.search(query , candidate_k , category=category)

        fused_scores: dict[str , float] = defaultdict(float)
        chunks: dict[str, DocumentChunk] = {}

        for results in (bm25_results, semantic_results):
            for rank, (chunk, _score) in enumerate(results, start=1):
                fused_scores[chunk.chunk_id] += (
                    1.0 / (self.rrf_k + rank)
                )
                chunks[chunk.chunk_id] = chunk

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                chunk_id,
            ),
        )

        return [
            (chunks[chunk_id], fused_scores[chunk_id])
            for chunk_id in ranked_ids[:top_k]
        ]


