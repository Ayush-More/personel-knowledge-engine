from pathlib import Path
import hashlib
from pathlib import Path

from knowledge_engine.models import Document

def get_document_category(filename: str) -> str:
    """Determine a document category from its filename."""
    name = Path(filename).name.lower()

    if name.startswith("project-"):
        return "projects"

    if name.startswith("work-experience-"):
        return "experience"

    categories = {
        "skills.md": "skills",
        "education.md": "education",
        "other-activities.md": "activities",
        "contact-information.md": "contact",
        "ayush-more-knowledge-base.md": "general",
    }

    return categories.get(name , "other")

def ingest_text_file(file_path:str | Path) -> Document:
    """Read a UTF-8 text file and convert it into a Document."""
    path = Path(file_path)

    content = path.read_text(encoding="utf-8").strip()

    normalized_path = str(path.resolve()).casefold()
    document_id = hashlib.sha256(
        normalized_path.encode("utf-8")
    ).hexdigest()

    return Document(
        document_id=document_id,
        content=content,
        source=str(path),
        metadata={
            "file_type": path.suffix.lower().lstrip("."),
            "category": get_document_category(path.name),
        }
    )