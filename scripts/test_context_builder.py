
from knowledge_engine.context_builder import build_context
from knowledge_engine.models import DocumentChunk


def main() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk-001",
        document_id="document-001",
        content="Developed an API to stream graphs in real time.",
        chunk_index=0,
        start_char=0,
        end_char=46,
    )

    context = build_context([(chunk, 0.92)])

    print("Generated context:")
    print(context)

    assert "chunk-001" in context
    assert "Developed an API" in context
    assert "Source 1" in context

    assert build_context([]) == ""

    print("\nAll context builder tests passed.")


if __name__ == "__main__":
    main()
