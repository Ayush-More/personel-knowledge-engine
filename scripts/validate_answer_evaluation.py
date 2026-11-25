
import json
from pathlib import Path

DATASET = Path("data/evaluation/answer_questions.jsonl")
CHUNKS = Path("data/processed/chunk.jsonl")

REQUIRED_FIELDS = {
    "question_id",
    "question",
    "expected_answer",
    "answerable",
    "evidence_chunk_ids",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main() -> None:
    questions = load_jsonl(DATASET)
    chunks = load_jsonl(CHUNKS)

    available_chunk_ids = {
        chunk["chunk_id"] for chunk in chunks
    }

    seen_ids = set()
    errors = []

    for line_number, item in enumerate(questions, start=1):
        missing = REQUIRED_FIELDS - item.keys()

        if missing:
            errors.append(
                f"Line {line_number}: missing fields {sorted(missing)}"
            )
            continue

        question_id = item["question_id"]

        if question_id in seen_ids:
            errors.append(
                f"Line {line_number}: duplicate ID {question_id}"
            )
        seen_ids.add(question_id)

        if not isinstance(item["answerable"], bool):
            errors.append(
                f"Line {line_number}: answerable must be true or false"
            )

        if not isinstance(item["evidence_chunk_ids"], list):
            errors.append(
                f"Line {line_number}: evidence_chunk_ids must be a list"
            )
            continue

        if item["answerable"] and not item["evidence_chunk_ids"]:
            errors.append(
                f"Line {line_number}: answerable question needs evidence"
            )

        if not item["answerable"] and item["evidence_chunk_ids"]:
            errors.append(
                f"Line {line_number}: unanswerable question should "
                "have no supporting evidence IDs"
            )

        for chunk_id in item["evidence_chunk_ids"]:
            if chunk_id not in available_chunk_ids:
                errors.append(
                    f"Line {line_number}: unknown chunk ID {chunk_id}"
                )

    if errors:
        print("Dataset validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Dataset validation passed: {len(questions)} questions.")
    print("All evidence chunk IDs exist in the processed chunks.")


if __name__ == "__main__":
    main()
