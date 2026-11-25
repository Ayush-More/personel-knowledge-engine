
import json
from pathlib import Path

from dotenv import load_dotenv

from knowledge_engine.answering import KnowledgeAnswer
from knowledge_engine.bm25 import BM25Retriever
from knowledge_engine.embeddings import EmbeddingModel
from knowledge_engine.gemini_generation import GeminiAnswerGenerator
from knowledge_engine.hybrid import HybridRetriever
from knowledge_engine.pipeline import RetrievalPipeline
from knowledge_engine.qdrant_retrieval import QdrantRetriever
from knowledge_engine.reranking import CrossEncoderReranker
from knowledge_engine.retrieval import load_chunks
from knowledge_engine.vector_store import VectorStore


DATASET_PATH = Path("data/evaluation/answer_questions.jsonl")
CHUNKS_PATH = Path("data/processed/chunk.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    load_dotenv()

    questions = load_jsonl(DATASET_PATH)
    chunks = load_chunks(str(CHUNKS_PATH))

    if not questions:
        raise ValueError("The answer-evaluation dataset is empty.")

    valid_chunk_ids = {chunk.chunk_id for chunk in chunks}

    store = VectorStore(
        storage_path="data/qdrant",
        collection_name="knowledge_chunks",
    )
    generator = None

    try:
        embedding_model = EmbeddingModel()

        semantic = QdrantRetriever(
            chunks=chunks,
            store=store,
            embedding_model=embedding_model,
        )
        semantic.index_chunks()

        hybrid = HybridRetriever(
            bm25=BM25Retriever(chunks),
            semantic=semantic,
        )

        pipeline = RetrievalPipeline(
            hybrid=hybrid,
            reranker=CrossEncoderReranker(),
            candidate_k=10,
        )

        generator = GeminiAnswerGenerator()
        answerer = KnowledgeAnswer(
            pipeline=pipeline,
            generator=generator,
        )

        answerability_correct = 0
        citation_count = 0
        valid_citation_count = 0
        answerable_count = 0
        evidence_hits = 0

        for item in questions:
            result = answerer.answer(item["question"], top_k=5)

            expected_answerable = item["answerable"]
            predicted_answerable = not result.insufficient_evidence

            if predicted_answerable == expected_answerable:
                answerability_correct += 1

            cited_ids = [citation.chunk_id for citation in result.citations]

            citation_count += len(cited_ids)
            valid_citation_count += sum(
                chunk_id in valid_chunk_ids for chunk_id in cited_ids
            )

            if expected_answerable:
                answerable_count += 1
                expected_ids = set(item["evidence_chunk_ids"])

                if expected_ids.intersection(cited_ids):
                    evidence_hits += 1

            print(f"\n[{item['question_id']}] {item['question']}")
            print(f"Expected answer: {item['expected_answer']}")
            print(f"Generated answer: {result.answer}")
            print(f"Expected answerable: {expected_answerable}")
            print(f"Predicted answerable: {predicted_answerable}")
            print(f"Citations: {cited_ids}")

        total = len(questions)

        print("\n========== Evaluation Summary ==========")
        print(f"Questions evaluated: {total}")
        print(
            f"Answerability accuracy: "
            f"{answerability_correct / total:.1%}"
        )

        if citation_count:
            print(
                f"Valid citation ID rate: "
                f"{valid_citation_count / citation_count:.1%}"
            )
        else:
            print("Valid citation ID rate: N/A (no citations)")

        if answerable_count:
            print(
                f"Evidence hit rate: "
                f"{evidence_hits / answerable_count:.1%}"
            )
        else:
            print("Evidence hit rate: N/A (no answerable questions)")

    finally:
        if generator is not None:
            generator.close()
        store.close()


if __name__ == "__main__":
    main()
