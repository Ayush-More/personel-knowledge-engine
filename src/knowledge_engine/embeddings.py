from sentence_transformers import SentenceTransformer

class EmbeddingModel:
     """Generate vector embeddings for text."""

     def __init__(self , model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
          self.model = SentenceTransformer(model_name)

     def encode(self , text: list[str]):
           """Convert a list of texts into numerical vectors."""
           return self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
           )