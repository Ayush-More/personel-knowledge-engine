
import json
from pathlib import Path

from knowledge_engine.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from knowledge_engine.retrieval import KeywordRetriever, load_chunks


def main() -> None:
    chunks_path = Path("data/processed/chunk.jsonl")
    questions_path = Path("data/evaluation/question.jsonl")
    k = 5

    chunks = load_chunks(chunks_path)
    retriever = KeywordRetriever(chunks)
    available_ids = {chunk.chunk_id for chunk in chunks}

    with questions_path.open("r", encoding="utf-8") as file:
        questions = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not questions:
        raise ValueError("The evaluation dataset is empty")

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    reciprocal_rank_scores: list[float] = []

    for question in questions:
        relevant_ids = set(question["relevant_chunk_ids"])

        if not relevant_ids:
            raise ValueError(
                f"{question['question_id']} has no relevant chunk IDs"
            )

        missing_ids = relevant_ids - available_ids
        if missing_ids:
            raise ValueError(
                f"{question['question_id']} references missing chunks: "
                f"{sorted(missing_ids)}. Regenerate chunks or update labels."
            )

        results = retriever.search(question["question"], top_k=k)
        retrieved_ids = [chunk.chunk_id for chunk, _ in results]

        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        recall = recall_at_k(retrieved_ids, relevant_ids, k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        precision_scores.append(precision)
        recall_scores.append(recall)
        reciprocal_rank_scores.append(rr)

        print(
            f"{question['question_id']} | "
            f"P@{k}={precision:.3f} | "
            f"R@{k}={recall:.3f} | "
            f"RR={rr:.3f}"
        )

    count = len(questions)
    print("\n=== Evaluation summary ===")
    print(f"Questions: {count}")
    print(f"Precision@{k}: {sum(precision_scores) / count:.3f}")
    print(f"Recall@{k}:    {sum(recall_scores) / count:.3f}")
    print(f"MRR:           {sum(reciprocal_rank_scores) / count:.3f}")


if __name__ == "__main__":
    main()
