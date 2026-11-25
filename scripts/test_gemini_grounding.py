
from dotenv import load_dotenv

from knowledge_engine.gemini_generation import GeminiAnswerGenerator
from knowledge_engine.models import DocumentChunk

load_dotenv()

chunk = DocumentChunk(
    chunk_id="grounding-test-001",
    document_id="grounding-test-doc",
    content=(
        "The application uses Python 3.12. "
        "It uses uv to manage dependencies and Qdrant "
        "to store vector embeddings locally."
    ),
    chunk_index=0,
    start_char=0,
    end_char=120,
)

generator = GeminiAnswerGenerator()

tests = [
    {
        "name": "Supported question",
        "question": "Which tool manages dependencies?",
        "results": [(chunk, 0.95)],
        "expect_insufficient": False,
    },
    {
        "name": "Unsupported question",
        "question": "Which database stores customer email addresses?",
        "results": [(chunk, 0.95)],
        "expect_insufficient": True,
    },
    {
        "name": "No retrieved evidence",
        "question": "What programming language does the app use?",
        "results": [],
        "expect_insufficient": True,
    },
]

try:
    for test in tests:
        result = generator.generate(
            question=test["question"],
            result=test["results"],
        )

        passed = (
            result.insufficient_evidence
            == test["expect_insufficient"]
        )

        print(f"\nTest: {test['name']}")
        print(f"Question: {result.question}")
        print(f"Answer: {result.answer}")
        print(f"Insufficient evidence: {result.insufficient_evidence}")
        print(f"Citations: {[c.chunk_id for c in result.citations]}")
        print(f"Status: {'PASS' if passed else 'FAIL'}")

finally:
    generator.close()
