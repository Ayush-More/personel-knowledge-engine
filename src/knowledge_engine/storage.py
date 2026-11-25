import hashlib
import json
from pathlib import Path

from knowledge_engine.chunking import DocumentChunk

def save_chunk_jsonl(
        chunks: list[DocumentChunk],
        output_path: str | Path
) -> None:
    """Save chunks as JSON Lines, one chunk per line."""
    path = Path(output_path)
    path.parent.mkdir(parents=True , exist_ok=True)

    with path.open("w" , encoding="utf-8") as file:
        for chunk in chunks:
            record = chunk.model_dump(mode="json")
            record["content_hash"] = hashlib.sha256(
                chunk.content.encode("utf-8")
            ).hexdigest()

            file.write(
                json.dumps(record , ensure_ascii=False) + "\n"
            )