from dotenv import load_dotenv
from knowledge_engine.answering import KnowledgeAnswer
from knowledge_engine.bm25 import BM25Retriever
from knowledge_engine.embeddings import EmbeddingModel
from knowledge_engine.generation import MockAnswerGeneration
from knowledge_engine.hybrid import HybridRetriever
from knowledge_engine.pipeline import RetrievalPipeline
from knowledge_engine.qdrant_retrieval import QdrantRetriever
from knowledge_engine.reranking import CrossEncoderReranker
from knowledge_engine.retrieval import load_chunks
from knowledge_engine.vector_store import VectorStore
from knowledge_engine.gemini_generation import GeminiAnswerGenerator

load_dotenv()
def main() -> None:
    chunks = load_chunks("data/processed/chunk.jsonl")
    embedding_model = EmbeddingModel()

    store = VectorStore(
        storage_path="data/qdrant",
        collection_name="knowledge_chunks",
    )

    try:
        semantic = QdrantRetriever(
            chunks=chunks,
            store=store,
            embedding_model=embedding_model,
        )
        semantic.index_chunks()

        hybrid = HybridRetriever(
            bm25=BM25Retriever(chunks),
            semantic=semantic,
        )

        retrieval_pipeline = RetrievalPipeline(
            hybrid=hybrid,
            reranker=CrossEncoderReranker(),
            candidate_k=10,
        )

        generator = GeminiAnswerGenerator()
        answerer = KnowledgeAnswer(
            pipeline=retrieval_pipeline,
            generator=generator,
        )

        result = answerer.answer(
            "What tools does this project use for semantic search?",
            top_k=5,
        )

        print("Question:", result.question)
        print("Answer:", result.answer)
        print("Insufficient evidence:", result.insufficient_evidence)
        print("Citations:")
        for citation in result.citations:
            print(f"- {citation.chunk_id}: {citation.text[:150]}")


        result = answerer.answer(
            "What API work did I do?"
        )

        print(result.model_dump_json(indent=2))

        assert result.question == "What API work did I do?"
        assert len(result.citations) > 0
        assert result.insufficient_evidence is False

        empty_question_error = False
        try:
            answerer.answer("   ")
        except ValueError:
            empty_question_error = True

        assert empty_question_error
        print("\nAll answering tests passed.")

    finally:
        store.close()


if __name__ == "__main__":
    main()
