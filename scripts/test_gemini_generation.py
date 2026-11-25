
from dotenv import load_dotenv

from knowledge_engine.gemini_generation import GeminiAnswerGenerator
from knowledge_engine.models import DocumentChunk

load_dotenv()

chunk = DocumentChunk(
    chunk_id="chunk-test-001",
    document_id="doc-test-001",
    content=(
        "The project uses Python 3.12 and uv for dependency "
        "management. Qdrant stores vector embeddings locally."
    ),
    chunk_index=0,
    start_char=0,
    end_char=100,
)

generator = GeminiAnswerGenerator()

try:
    result = generator.generate(
        question="Which tools does the project use for Python dependencies "
                 "and vector storage?",
        result=[(chunk, 0.95)],
    )

    print("Answer:", result.answer)
    print("Citations:", [c.chunk_id for c in result.citations])
    print("Insufficient evidence:", result.insufficient_evidence)

finally:
    generator.close()
