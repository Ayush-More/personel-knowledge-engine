
from knowledge_engine.embeddings import EmbeddingModel
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
        # Embed and store each knowledge chunk.
        vectors = model.encode([chunk.content for chunk in chunks])

        for chunk, vector in zip(chunks, vectors):
            store.upsert(
                chunk_id=chunk.chunk_id,
                text=chunk.content,
                vector=vector.tolist(),
            )

        query = "How can I improve performance?"
        query_vector = model.encode([query])[0]

        results = store.search(query_vector.tolist(), limit=5)

        print(f"Chunks loaded: {len(chunks)}")
        print(f"Results returned: {len(results)}")

        for rank, result in enumerate(results, start=1):
            preview = " ".join(result["text"].split())[:180]
            print(f"\n{rank}. Score: {result['score']:.4f}")
            print(f"   Chunk ID: {result['chunk_id']}")
            print(f"   Text: {preview}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
