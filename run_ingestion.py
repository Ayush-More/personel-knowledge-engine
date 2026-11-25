from src.knowledge_engine.ingestion import ingest_text_file

document =  ingest_text_file("data/notes.txt")

print("Document ID" , document.document_id)
print("Content" , document.content)
print("Metadata" , document.metadata)
print("Source" , document.source)
print("Created AT" , document.created_at)