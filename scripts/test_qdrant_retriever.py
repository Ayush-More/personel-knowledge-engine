
from knowledge_engine.embeddings import EmbeddingModel
from knowledge_engine.qdrant_retrieval import QdrantRetriever
from knowledge_engine.retrieval import load_chunks
from knowledge_engine.vector_store import VectorStore


def main() -> None:
    chunks = load_chunks("data/processed/chunk.jsonl")
    model = EmbeddingModel()

    store = VectorStore(
        storage_path="data/qdrant",
        collection_name="knowledge_chunks",
    )

    try:
        retriever = QdrantRetriever(
            chunks=chunks,
            store=store,
            embedding_model=model,
        )

        # For this first test, make sure every chunk is indexed.
        retriever.index_chunks()

        query = "How can I improve performance?"
        results = retriever.search(query, top_k=5)

        print(f"Chunks loaded: {len(chunks)}")
        print(f"Results returned: {len(results)}")

        for rank, (chunk, score) in enumerate(results, start=1):
            preview = " ".join(chunk.content.split())[:180]
            print(f"\n{rank}. Score: {score:.4f}")
            print(f"   Chunk ID: {chunk.chunk_id}")
            print(f"   Text: {preview}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
