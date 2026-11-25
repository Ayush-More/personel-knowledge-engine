import math
from collections import Counter

from knowledge_engine.models import DocumentChunk
from knowledge_engine.retrieval import tokenize

class BM25Retriever:
    """A simple BM25 retriever for a collection of chunks."""

    def __init__(self , chunks: list[DocumentChunk] , k1: float = 1.75 , b: float = 0.5):
        if k1<0:
            raise ValueError("k1 must be non negative")

        if not 0<=b<=1:
            raise ValueError("b must be between 0 and 1")

        self.chunks = {
            chunk.chunk_id: chunk for chunk in chunks
        }

        self.k1 = k1
        self.b = b

        self.tokens = {
            chunk.chunk_id: tokenize(chunk.content) for chunk in chunks
        }

        self.term_frequencies = {
            chunk_id: Counter(tokens) for chunk_id , tokens in self.tokens.items()
        }

        self.document_count = len(chunks)

        self.document_lengths = {
            chunk_id: len(tokens) for chunk_id , tokens in self.tokens.items()
        }

        self.average_length = (
            sum(self.document_lengths.values())/self.document_count
            if self.document_count else 0.0
        )

        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens.values():
            self.document_frequency.update(set(tokens))

    def search(self , query:str , top_k: int = 5 , category: str | None = None) -> list[tuple[DocumentChunk , float]]:
         """Return chunks ranked by their BM25 score."""

         if top_k <= 0:
             raise ValueError("top_k must be greater then or equal to zero")
         if self.document_count == 0:
             return []

         query_terms = set(tokenize(query))
         scores: dict[str , float] = {}

         for chunk_id , term_frequencies in self.term_frequencies.items():
             chunk = self.chunks[chunk_id]

             if category is not None and chunk.metadata.get("category") != category:
                 continue

             score = 0.0
             doc_length = self.document_lengths[chunk_id]

             for term in query_terms:
                 tf = term_frequencies.get(term , 0)
                 if tf == 0:
                     continue

                 df = self.document_frequency[term]

                   # Smoothed inverse document frequency.
                 idf = math.log(
                    1 + (self.document_count - df + 0.5) / (df + 0.5)
                )

                 length_norm = (
                    1 - self.b + self.b * doc_length / self.average_length
                ) if self.average_length else 1.0

                 tf_weight = (
                    tf * (self.k1 + 1)
                    / (tf + self.k1 * length_norm)
                )

                 score += idf * tf_weight

             if score > 0:
                scores[chunk_id] = score

         ranked_ids = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], chunk_id),
        )

         return [
            (self.chunks[chunk_id], scores[chunk_id])
            for chunk_id in ranked_ids[:top_k]
        ]

