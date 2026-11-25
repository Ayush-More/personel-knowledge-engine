
from knowledge_engine.embeddings import EmbeddingModel


def main() -> None:
    model = EmbeddingModel()

    texts = [
        "How can I make my application faster?",
        "Ways to improve software performance.",
        "My favorite food is chocolate cake.",
    ]

    vectors = model.encode(texts)

    print("Number of texts:", len(texts))
    print("Vector matrix shape:", vectors.shape)
    print("First vector dimensions:", vectors[0].shape)
    print("First five values:", vectors[0][:5])


if __name__ == "__main__":
    main()
