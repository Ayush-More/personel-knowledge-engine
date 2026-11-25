from knowledge_engine.models import RetrievedEvidence, ToolResult
from knowledge_engine.pipeline import RetrievalPipeline

class KnowledgeTools:
    """Expose knowledge retrieval capabilities as agent tools."""

    def __init__(self , pipeline: RetrievalPipeline)-> None:
        self.pipeline = pipeline

    def search_projects(
        self,
        query: str,
        top_k: int = 5,
    ) -> ToolResult:
        """Search project-related knowledge."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        results = self.pipeline.search(
            query=query,
            top_k=top_k,
            category="projects",
        )

        evidence = [
            RetrievedEvidence(
                document_id = chunk.document_id,
                chunk_index = chunk.chunk_index,
                start_char = chunk.start_char,
                end_char= chunk.end_char, 
                metadata= chunk.metadata,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source=chunk.source,
                score=float(score),
            )
            for chunk, score in results
        ]

        return ToolResult(
            tool_name="search_projects",
            query=query,
            evidence=evidence,
            result_count=len(evidence),
        )

    def search_skills(self, query: str, top_k: int = 5) -> ToolResult:
        """Search skill-related knowledge."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        results = self.pipeline.search(
            query=query,
            top_k=top_k,
            category="skills",
        )

        evidence = [
            RetrievedEvidence(
                document_id = chunk.document_id,
                chunk_index = chunk.chunk_index,
                start_char = chunk.start_char,
                end_char= chunk.end_char, 
                metadata= chunk.metadata,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source=chunk.source,
                score=float(score),
            )
            for chunk, score in results
        ]

        return ToolResult(
            tool_name="search_skills",
            query=query,
            evidence=evidence,
            result_count=len(evidence),
        )

    def search_category(self , query: str = "" , top_k: int = 5 , tool_name= str , category=str )-> ToolResult:
                """Search a category and convert results into structured evidence."""

                if not query.strip():
                    raise ValueError("Query must not be empty")
        
                if top_k<=0:
                    raise ValueError("top_k must be greater than zero")
        
                results = self.pipeline.search(
                    query=query,
                    top_k=top_k,
                    category=category
                )
        
                evidence = [
                    RetrievedEvidence(
                        document_id = chunk.document_id,
                        chunk_index = chunk.chunk_index,
                        start_char = chunk.start_char,
                        end_char= chunk.end_char, 
                        metadata= chunk.metadata,
                        chunk_id = chunk.chunk_id,
                        content = chunk.content,
                        source = chunk.source,
                        score = float(score),
                    ) for chunk , score in results
                ]
        
                return ToolResult(
                    tool_name=tool_name,
                    query=query,
                    evidence=evidence,
                    result_count=len(evidence)
                )

    