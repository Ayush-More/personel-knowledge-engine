from sentence_transformers import CrossEncoder

from knowledge_engine.models import DocumentChunk

class CrossEncoderReranker:
    """Rerank retrieved chunks using query–chunk relevance."""

    def __init__(self , model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self , query: str , candidates: list[tuple[DocumentChunk , float]], top_k: int = 5) -> list[tuple[DocumentChunk , float]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
            
        if not query.strip():
            return []

        pairs = [
            (query, chunk.content)
            for chunk, _original_score in candidates
        ]

        scores = self.model.predict(pairs)

        reranked = [
            (chunk, float(score))
            for (chunk, _original_score), score in zip(candidates, scores)
        ]

        reranked.sort(
            key=lambda item: (-item[1], item[0].chunk_id)
        )

        return reranked[:top_k]