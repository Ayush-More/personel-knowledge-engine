import argparse
from pathlib import Path

from knowledge_engine.chunking import chunk_document_by_paragraph
from knowledge_engine.ingestion import ingest_text_file
from knowledge_engine.storage import save_chunk_jsonl


def ingest_files(
        input_files: list[Path],
        chunk_size: int
)-> list:
     """Ingest multiple files and combine their chunks."""
     all_chunks = []

     for input_file in input_files:
         document = ingest_text_file(input_file)

         chunks = chunk_document_by_paragraph(document=document , chunk_size=chunk_size)

         all_chunks.extend(chunks)

     return all_chunks

def main() -> None:
    parse = argparse.ArgumentParser(
        description="Ingest a text file and save its chunk"
    )

    parse.add_argument(
        "--input",
        type=Path,
        default=Path("data/ayush-more-knowledge-base.md"),
        help="Path to source text file"
    )

    parse.add_argument(
        "--all",
        action="store_true",
        help="Ingest all mardown file in data directory"
    )


    parse.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunk.jsonl"),
        help="Path for jsonl Output"
    )

    parse.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum character per chunk"
    )

    args = parse.parse_args()

    if args.all:
        input_files = sorted(
            Path("data").glob("*.md")
        )
    else:
        input_files = [args.input]

    chunks = ingest_files(
        input_files = input_files,
        chunk_size=args.chunk_size,
    )
    save_chunk_jsonl(chunks , args.output)

    print(f"Documents processed: {len(input_files)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved to: {args.output}")



if __name__ == "__main__":
    main()