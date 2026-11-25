from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException

from knowledge_engine.answering import KnowledgeAnswer
from knowledge_engine.bm25 import BM25Retriever
from knowledge_engine.embeddings import EmbeddingModel
from knowledge_engine.gemini_generation import GeminiAnswerGenerator
from knowledge_engine.hybrid import HybridRetriever
from knowledge_engine.pipeline import RetrievalPipeline
from knowledge_engine.qdrant_retrieval import QdrantRetriever
from knowledge_engine.reranking import CrossEncoderReranker
from knowledge_engine.retrieval import load_chunks
from knowledge_engine.vector_store import VectorStore

from knowledge_engine.api.schemas import AskRequest , AskResponse , CitationResponse

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG engine when the API starts."""

    chunks = load_chunks("data/processed/chunk.jsonl")

    embedding_model = EmbeddingModel()

    store = VectorStore(
        storage_path="data/qdrant",
        collection_name="knowledge_chunks"
    )

    generator = GeminiAnswerGenerator()

    try:
        semantic = QdrantRetriever(
            chunks=chunks,
            store=store,
            embedding_model=embedding_model
        ) 

        semantic.index_chunks()

        hybrid = HybridRetriever(
            bm25=BM25Retriever(chunks),
            semantic=semantic
        )

        pipeline = RetrievalPipeline(
            hybrid=hybrid,
            reranker=CrossEncoderReranker(),
            candidate_k=10
        )

        app.state.answerer = KnowledgeAnswer(
            pipeline=pipeline,
            generator=generator,
        )


        yield
    finally:
        generator.close()
        store.close()

app =  FastAPI(
    title="Personal Knowledge Engine",
    description="A RAG powered personal knowledge assistant",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check():
    return {"status":"healthy"}

@app.post("/ask" , response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        result = app.state.answerer.answer(
            request.question,
            top_k=request.top_k
        )

        return AskResponse(
            question=result.question,
            answer=result.answer,
            citation=[
                CitationResponse(
                    chunk_id=citation.chunk_id,
                    text= citation.text
                )
                for citation in result.citations
            ],
            insufficient_evidence=result.insufficient_evidence
        )
    except Exception:
        logger.exception("Failed to process question")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your question.",
        )