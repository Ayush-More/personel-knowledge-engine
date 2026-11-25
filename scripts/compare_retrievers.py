
import json
from pathlib import Path

from knowledge_engine.bm25 import BM25Retriever
from knowledge_engine.embeddings import EmbeddingModel
from knowledge_engine.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from knowledge_engine.hybrid import HybridRetriever
from knowledge_engine.qdrant_retrieval import QdrantRetriever
from knowledge_engine.reranking import CrossEncoderReranker
from knowledge_engine.retrieval import KeywordRetriever, load_chunks
from knowledge_engine.semantic import SemanticRetriever
from knowledge_engine.vector_store import VectorStore


def main() -> None:
    # 1. Load chunks and evaluation questions
    chunks = load_chunks("data/processed/chunk.jsonl")

    with Path("data/evaluation/question.jsonl").open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not chunks:
        raise ValueError("The chunk dataset is empty")

    if not questions:
        raise ValueError("The evaluation dataset is empty")

    available_ids = {chunk.chunk_id for chunk in chunks}

    # 2. Initialize models and retrievers
    print("Loading embedding model...")
    embedding_model = EmbeddingModel()

    keyword = KeywordRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    semantic = SemanticRetriever(chunks, embedding_model)

    # 3. Initialize and index Qdrant
    store = VectorStore(
        storage_path="data/qdrant",
        collection_name="knowledge_chunks",
    )

    semantic_qdrant = QdrantRetriever(
        chunks=chunks,
        store=store,
        embedding_model=embedding_model,
    )

    print("Indexing chunks into Qdrant...")
    semantic_qdrant.index_chunks()

    # 4. Initialize hybrid retrieval and reranking
    hybrid = HybridRetriever(
        bm25=bm25,
        semantic=semantic_qdrant,
    )

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker()

    # 5. Define the retrieval configurations to evaluate
    retrievers = {
        "Keyword": keyword,
        "BM25": bm25,
        "Semantic": semantic,
        "Qdrant Semantic": semantic_qdrant,
        "Hybrid": hybrid,
    }

    k = 5
    candidate_k = 10

    # Store metrics for each configuration
    metrics: dict[str, dict[str, float]] = {}

    # 6. Evaluate each retrieval configuration
    for name, retriever in retrievers.items():
        precision_scores: list[float] = []
        recall_scores: list[float] = []
        rr_scores: list[float] = []

        print(f"\nEvaluating {name}...")

        for question in questions:
            query = question["question"]
            relevant_ids = set(question["relevant_chunk_ids"])

            if not relevant_ids:
                raise ValueError(
                    f"{question['question_id']} has no relevant chunks"
                )

            missing_ids = relevant_ids - available_ids
            if missing_ids:
                raise ValueError(
                    f"{question['question_id']} references missing chunks: "
                    f"{sorted(missing_ids)}"
                )

            # Retrieve top-k results
            results = retriever.search(query, top_k=k)

            retrieved_ids = [
                chunk.chunk_id for chunk, _ in results
            ]

            precision_scores.append(
                precision_at_k(retrieved_ids, relevant_ids, k)
            )
            recall_scores.append(
                recall_at_k(retrieved_ids, relevant_ids, k)
            )
            rr_scores.append(
                reciprocal_rank(retrieved_ids, relevant_ids)
            )

        count = len(questions)

        metrics[name] = {
            "Precision@5": sum(precision_scores) / count,
            "Recall@5": sum(recall_scores) / count,
            "MRR": sum(rr_scores) / count,
        }

    # 7. Evaluate hybrid retrieval followed by reranking
    print("\nEvaluating Hybrid + Reranker...")

    precision_scores = []
    recall_scores = []
    rr_scores = []

    for question in questions:
        query = question["question"]
        relevant_ids = set(question["relevant_chunk_ids"])

        # Retrieve more candidates than the final number needed
        candidates = hybrid.search(query, top_k=candidate_k)

        # Rerank candidates and keep the best k
        reranked_results = reranker.rerank(
            query,
            candidates,
            top_k=k,
        )

        retrieved_ids = [
            chunk.chunk_id for chunk, _ in reranked_results
        ]

        precision_scores.append(
            precision_at_k(retrieved_ids, relevant_ids, k)
        )
        recall_scores.append(
            recall_at_k(retrieved_ids, relevant_ids, k)
        )
        rr_scores.append(
            reciprocal_rank(retrieved_ids, relevant_ids)
        )

    count = len(questions)

    metrics["Hybrid + Reranker"] = {
        "Precision@5": sum(precision_scores) / count,
        "Recall@5": sum(recall_scores) / count,
        "MRR": sum(rr_scores) / count,
    }

    # 8. Print comparison table
    print("\n" + "=" * 72)
    print("RETRIEVER EVALUATION RESULTS")
    print("=" * 72)
    print(f"Evaluation questions: {len(questions)}")
    print(f"Final result count (k): {k}")
    print(f"Hybrid candidate count: {candidate_k}")
    print("-" * 72)

    print(
        f"{'Retriever':<24}"
        f"{'Precision@5':>15}"
        f"{'Recall@5':>13}"
        f"{'MRR':>10}"
    )
    print("-" * 72)

    for name, scores in metrics.items():
        print(
            f"{name:<24}"
            f"{scores['Precision@5']:>15.3f}"
            f"{scores['Recall@5']:>13.3f}"
            f"{scores['MRR']:>10.3f}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
