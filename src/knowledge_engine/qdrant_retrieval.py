
from knowledge_engine.embeddings import EmbeddingModel
from knowledge_engine.models import DocumentChunk
from knowledge_engine.retrieval import load_chunks
from knowledge_engine.vector_store import VectorStore


class QdrantRetriever:
    """Retrieve knowledge chunks using Qdrant vector search."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        store: VectorStore,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.chunks = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }
        self.store = store
        self.embedding_model = embedding_model or EmbeddingModel()

    def index_chunks(self) -> None:
        """Embed chunks and store their vectors in Qdrant."""
        chunks = list(self.chunks.values())

        if not chunks:
            return

        vectors = self.embedding_model.encode(
            [chunk.content for chunk in chunks]
        )

        for chunk, vector in zip(chunks, vectors):
            self.store.upsert(
                chunk_id=chunk.chunk_id,
                text=chunk.content,
                vector=vector.tolist(),
                source=chunk.source,
                metadata=chunk.metadata
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return Qdrant results in the standard retriever format."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not query.strip():
            return []

        query_vector = self.embedding_model.encode([query])[0]

        results = self.store.search(
            query_vector.tolist(),
            limit=top_k,
            category=category,
        )

        ranked_chunks: list[tuple[DocumentChunk, float]] = []

        for result in results:
            chunk_id = result["chunk_id"]
            chunk = self.chunks.get(chunk_id)

            if chunk is not None:
                ranked_chunks.append((chunk, float(result["score"])))

        return ranked_chunks
