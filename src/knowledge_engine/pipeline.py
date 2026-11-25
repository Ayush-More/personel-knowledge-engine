from knowledge_engine.hybrid import HybridRetriever
from knowledge_engine.models import DocumentChunk
from knowledge_engine.reranking import CrossEncoder

class RetrievalPipeline:
    """Retrieve candidate chunks, then rerank them."""

    def __init__(self , hybrid: HybridRetriever , reranker: CrossEncoder , candidate_k:int = 10)->None:
        if candidate_k <= 0:
            raise ValueError("Cnadidate_k must be greater then zero")

        self.hybrid = hybrid
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(self , query: str , top_k: int = 5 , category: str | None = None)-> list[tuple[DocumentChunk , float]]:
        if top_k<=0:
            raise ValueError("top_k must be greater then zero")

        if not query.strip():
            return []

        candidates = self.hybrid.search(
            query,
            top_k=max(self.candidate_k , top_k),
            category=category
        )

        return self.reranker.rerank(
            query , candidates , top_k=top_k
        )