import numpy as np

from knowledge_engine.embeddings import EmbeddingModel
from knowledge_engine.similarity import cosine_similarity
from knowledge_engine.models import DocumentChunk

class SemanticRetriever:
    """Retrieve knowledge chunks using embedding similarity."""

    def __init__(self , chunks: list[DocumentChunk] , embedding_model: EmbeddingModel | None = None):
        self.chunks = chunks
        self.embedding_model = embedding_model or EmbeddingModel()

        texts = [chunk.content for chunk in chunks]

        if texts:
            self.embeddings = self.embedding_model.encode(texts)
        else:
            self.embeddings = np.empty((0,0) , dtype=np.float32)

    def search(self , query : str , top_k: int = 5)-> list[tuple[DocumentChunk , float]]:
        """Return chunks ranked by cosine similarity."""

        if top_k <=0:
            raise ValueError("top_k cannot be smaller then zero")

        if not query.strip() or not self.chunks:
            return []

        query_embedding = self.embedding_model.encode([query])[0]

        scored_result = [
            (
                chunk,
                cosine_similarity(query_embedding , chunk_embedding)
            )
            for chunk , chunk_embedding in zip(
                self.chunks,
                self.embeddings,
            )
        ]

        scored_result.sort(key= lambda item: (-item[1] , item[0].chunk_id))

        return scored_result[:top_k]