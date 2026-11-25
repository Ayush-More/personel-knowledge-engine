
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models


class VectorStore:
    """Persist and search vectors using local Qdrant storage."""

    def __init__(
        self,
        storage_path: str | Path = "data/qdrant",
        collection_name: str = "knowledge_chunks",
        vector_size: int = 384,
    ) -> None:
        self.client = QdrantClient(path=str(storage_path))
        self.collection_name = collection_name
        self.vector_size = vector_size

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    @staticmethod
    def point_id(chunk_id: str) -> str:
        """Convert a chunk ID into a deterministic UUID."""
        return str(uuid5(NAMESPACE_URL, chunk_id))

    def upsert(
        self,
        chunk_id: str,
        text: str,
        vector: list[float],
        source: str = "",
        metadata: dict[str , str] | None = None
    ) -> None:
        """Insert or update one chunk and its embedding."""
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Expected vector size {self.vector_size}, got {len(vector)}"
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=self.point_id(chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,
                        "text": text,
                        "source": source,
                        "metadata": metadata or {}
                    },
                )
            ],
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        category: str | None = None,
    ) -> list[dict]:
        """Return the most similar stored chunks."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Expected vector size {self.vector_size}, "
                f"got {len(query_vector)}"
            )

        query_filter = None

        if category is not None:
            query_filter = models.Filter(
                must = [
                    models.FieldCondition(
                        key="metadata.category",
                        match=models.MatchValue(value=category)
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points

        return [
            {
                "chunk_id": point.payload["chunk_id"],
                "text": point.payload["text"],
                "score": point.score,
                "source": point.payload.get("source", ""),
                "metadata": point.payload.get("metadata", {}),
            }
            for point in results
        ]

    def close(self) -> None:
        """Close the local Qdrant database."""
        self.client.close()
