import numpy as np

def cosine_similarity(
        vector_a: np.ndarray,
        vector_b: np.ndarray
)->float:
     """Calculate cosine similarity between two vectors."""
     if vector_a.ndim != 1 or vector_b.ndim != 1:
          raise ValueError("Both inputs must be one-dimensional vectors")

     if vector_a.shape != vector_b.shape:
          raise ValueError("Both vector must be of same dimension")

     norm_a = np.linalg.norm(vector_a)
     norm_b = np.linalg.norm(vector_b)

     if norm_a == 0 or norm_b == 0:
          raise ValueError("Cosine similarity is undefined for zero vectors")


     return float(np.dot(vector_a , vector_b)/(norm_a * norm_b))