import json
import re
from collections import Counter
from pathlib import Path

from knowledge_engine.models import DocumentChunk

def tokenize(text : str) -> list[str]:
     """Convert text into lowercase word tokens."""
     return re.findall(r"\b\w+\b" , text.casefold())

def load_chunks(path : str | Path) -> list[DocumentChunk]:
    """Load chunks from a JSONL file."""
    chunks : list[DocumentChunk] = []

    with Path(path).open("r" , encoding="utf-8") as file:
          for line in file:
               if not line.strip():
                    continue

               record = json.loads(line)

               record.pop("content_hash",None)
               chunks.append(DocumentChunk.model_validate(record))


    return chunks

class KeywordRetriever:
    """A basic inverted-index keyword retriever."""

    def __init__(self , chunks : list[DocumentChunk]) -> None:
        self.chunks = {
             chunk.chunk_id: chunk
             for chunk in chunks
        }

        self.index: dict[str, set[str]] = {}

        for chunk in chunks:
             for term in set(tokenize(chunk.content)):
                self.index.setdefault(term , set()).add(chunk.chunk_id)

    def search(self , query : str , top_k: int = 5) -> list[tuple[DocumentChunk , float]]:
         """Rank chunks by the number of distinct query terms matched."""

         if top_k <= 0:
              raise ValueError("top_k must be greate then 0")

         scores : Counter[str] = Counter()

         for term in set(tokenize(query)):
              for chunk_id in self.index.get(term , set()):
                   scores[chunk_id] += 1

         ranked_ids = sorted(
            scores,
            key=lambda chunk_id: (
                -scores[chunk_id],
                chunk_id,
            ),
        )

         return [
            (self.chunks[chunk_id], float(scores[chunk_id]))
            for chunk_id in ranked_ids[:top_k]
        ]