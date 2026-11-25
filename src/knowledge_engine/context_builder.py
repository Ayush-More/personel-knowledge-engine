from knowledge_engine.models import DocumentChunk

def build_context(result: list[tuple[DocumentChunk , float]])->str:
     """Format retrieved chunks into a context string for an LLM."""
     if not result:
          return ""

     sections = []

     for rank , (chunk , score) in enumerate(result , start=1):
          sections.append(
               f"[Source {rank} | Chunk ID: {chunk.chunk_id} |"
               f"Retrival score : {score:.4f}\n"
               f"{chunk.content}"
          ) 

     return "\n\n---\n\n".join(sections)