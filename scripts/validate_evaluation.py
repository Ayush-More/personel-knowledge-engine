
import json
from pathlib import Path

path = Path("data/evaluation/question.jsonl")

required_fields = {
    "question_id",
    "question",
    "expected_answer",
    "relevant_chunk_ids",
}

seen_ids: set[str] = set()

with path.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        record = json.loads(line)

        missing = required_fields - record.keys()
        if missing:
            raise ValueError(
                f"Line {line_number} is missing: {sorted(missing)}"
            )

        if record["question_id"] in seen_ids:
            raise ValueError(
                f"Duplicate question ID: {record['question_id']}"
            )

        if not record["relevant_chunk_ids"]:
            raise ValueError(
                f"Line {line_number} needs a relevant chunk ID"
            )

        seen_ids.add(record["question_id"])

print(f"Valid evaluation questions: {len(seen_ids)}")
