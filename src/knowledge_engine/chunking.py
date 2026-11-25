from knowledge_engine.models import DocumentChunk , Document
from knowledge_engine.config import settings
import re


def chunk_document(document: Document , 
                   chunk_size: int | None = None , 
                   chunk_overlap: int | None = None) -> list[DocumentChunk]:

    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = (
        chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    )

    if size <= 0:
        raise ValueError("chunk size must be greater then zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= size:
        raise ValueError("Overlap cannot be greater then size")

    text = document.content
    chunks: list[DocumentChunk] = []
    start = 0

    while start < len(text):
        end = min(start + size , len(text))
        content = text[start:end].strip()

        if content:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}:fixed:{len(chunks)}",
                    document_id= document.document_id,
                    content=content,
                    chunk_index= len(chunks),
                    start_char= start,
                    end_char= end,
                    source=document.source,
                    metadata=document.metadata.copy()
                )
            )

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_document_by_paragraph(
    document: Document,
    chunk_size: int | None = None,
    chunk_overlap: int = 0,
) -> list[DocumentChunk]:
    """Group whole paragraphs into chunks without exceeding chunk_size.

    Paragraphs longer than chunk_size are split using the existing
    fixed-size chunker.
    """
    size = chunk_size if chunk_size is not None else settings.chunk_size

    if size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap != 0:
        raise ValueError(
            "Paragraph chunking currently supports chunk_overlap=0 only"
        )

    text = document.content

    paragraphs = [
    (match.start(), match.group())
    for match in re.finditer(
        r"\S(?:.*?\S)?(?=\n[ \t]*\n|\Z)",
        text,
        flags=re.DOTALL,
    )
    ]

    chunks: list[DocumentChunk] = []
    pending: list[tuple[int, str]] = []

    def emit_pending() -> None:
        if not pending:
            return

        start = pending[0][0]
        end = pending[-1][0] + len(pending[-1][1])
        content = "\n\n".join(p for _, p in pending).strip()

        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.document_id}:chunk:{len(chunks)}",
                document_id=document.document_id,
                content=content,
                chunk_index=len(chunks),
                start_char=start,
                end_char=end,
                source=document.source,
                metadata=document.metadata.copy()
            )
        )
        pending.clear()

    for paragraph_start, paragraph in paragraphs:
        if len(paragraph) > size:
            emit_pending()

            temporary_document = Document(
                document_id=document.document_id,
                content=paragraph,
                source=document.source,
            )
            pieces = chunk_document(
                temporary_document,
                chunk_size=size,
                chunk_overlap=0,
            )

            for piece in pieces:
                chunks.append(
                    DocumentChunk(
                        chunk_id=(
                            f"{document.document_id}:chunk:{len(chunks)}"
                        ),
                        document_id=document.document_id,
                        content=piece.content,
                        chunk_index=len(chunks),
                        start_char=paragraph_start + piece.start_char,
                        end_char=paragraph_start + piece.end_char,
                        source=document.source,
                        metadata=document.metadata.copy()
                    )
                )
            continue

        pending_length = sum(len(p) for _, p in pending)
        separator_length = 2 * len(pending)

        if pending and pending_length + separator_length + len(paragraph) > size:
            emit_pending()

        pending.append((paragraph_start, paragraph))

    emit_pending()
    return chunks
