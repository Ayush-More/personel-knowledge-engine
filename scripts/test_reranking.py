
from knowledge_engine.bm25 import BM25Retriever
from knowledge_engine.embeddings import EmbeddingModel
from knowledge_engine.hybrid import HybridRetriever
from knowledge_engine.models import DocumentChunk
from knowledge_engine.qdrant_retrieval import QdrantRetriever
from knowledge_engine.reranking import CrossEncoderReranker
from knowledge_engine.retrieval import load_chunks
from knowledge_engine.vector_store import VectorStore


def main() -> None:
    chunks = load_chunks("data/processed/chunk.jsonl")
    embedding_model = EmbeddingModel()

    store = VectorStore(
        storage_path="data/qdrant",
        collection_name="knowledge_chunks",
    )

    try:
        semantic = QdrantRetriever(
            chunks=chunks,
            store=store,
            embedding_model=embedding_model,
        )
        semantic.index_chunks()

        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(bm25=bm25, semantic=semantic)

        query = "How can I improve performance?"
        candidates = hybrid.search(query, top_k=10)

        reranker = CrossEncoderReranker()
        results = reranker.rerank(
            query,
            candidates,
            top_k=5,
        )

        print(f"Query: {query}")
        print(f"Candidates before reranking: {len(candidates)}")
        print(f"Results after reranking: {len(results)}")

        for rank, (chunk, score) in enumerate(results, start=1):
            preview = " ".join(chunk.content.split())[:180]
            print(f"\n{rank}. Reranker score: {score:.4f}")
            print(f"   Chunk ID: {chunk.chunk_id}")
            print(f"   Text: {preview}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
