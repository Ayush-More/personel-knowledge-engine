
from knowledge_engine.embeddings import EmbeddingModel
from knowledge_engine.similarity import cosine_similarity


def main() -> None:
    model = EmbeddingModel()

    texts = [
        "How can I make my application faster?",
        "Ways to improve software performance.",
        "My favorite food is chocolate cake.",
    ]

    vectors = model.encode(texts)

    similar_score = cosine_similarity(vectors[0], vectors[1])
    unrelated_score = cosine_similarity(vectors[0], vectors[2])

    print(f"Performance sentence similarity: {similar_score:.4f}")
    print(f"Unrelated sentence similarity:   {unrelated_score:.4f}")

    assert similar_score > unrelated_score, (
        "Expected the performance-related sentences to be more similar"
    )

    print("Similarity test passed!")


if __name__ == "__main__":
    main()
