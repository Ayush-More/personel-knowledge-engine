
from knowledge_engine.retrieval import load_chunks
from knowledge_engine.semantic import SemanticRetriever


def main() -> None:
    chunks = load_chunks("data/processed/chunk.jsonl")
    retriever = SemanticRetriever(chunks)

    query = "How can I improve performance?"
    results = retriever.search(query, top_k=5)

    print(f"Query: {query}")
    print(f"Chunks available: {len(chunks)}")
    print(f"Results returned: {len(results)}")

    for rank, (chunk, score) in enumerate(results, start=1):
        preview = " ".join(chunk.content.split())[:180]
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   Chunk ID: {chunk.chunk_id}")
        print(f"   Text: {preview}")


if __name__ == "__main__":
    main()
