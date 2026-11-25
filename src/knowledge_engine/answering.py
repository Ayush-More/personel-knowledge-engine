
from knowledge_engine.generation import MockAnswerGeneration
from knowledge_engine.models import (
    GeneratedAnswer,
    AgentRunResult,
    DocumentChunk,
)
from knowledge_engine.pipeline import RetrievalPipeline
from knowledge_engine.agent.runner import AgentRunner


class KnowledgeAnswer:
    """Connect retrieval, agent execution, and answer generation."""

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        generator=None,
        agent_runner: AgentRunner | None = None,
    ):
        self.pipeline = pipeline
        self.generator = generator or MockAnswerGeneration
        self.agent_runner = agent_runner

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("Question must not be empty")

        results = self.pipeline.search(
            query=question,
            top_k=top_k,
        )

        return self.generator.generate(
            question=question,
            result=results,
        )

    def answer_from_agent(
        self,
        question: str,
        agent_result: AgentRunResult,
    ) -> GeneratedAnswer:
        """Generate a grounded answer from evidence collected by the agent."""

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if agent_result.status == "failed":
            raise RuntimeError(
                "Cannot generate an answer from a failed agent run"
            )

        results: list[tuple[DocumentChunk, float]] = []

        for evidence in agent_result.evidence:
            chunk = DocumentChunk(
                chunk_id=evidence.chunk_id,
                document_id=evidence.document_id,
                content=evidence.content,
                chunk_index=evidence.chunk_index,
                start_char=evidence.start_char,
                end_char=evidence.end_char,
                source=evidence.source,
                metadata=evidence.metadata,
            )

            results.append((chunk, evidence.score))

        return self.generator.generate(
            question=question,
            result=results,
        )

    def answer_with_agent(self, question: str) -> GeneratedAnswer:
        """Run the agent and generate an answer from its collected evidence."""

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if self.agent_runner is None:
            raise RuntimeError("AgentRunner is not configured")

        agent_result = self.agent_runner.run(question)

        return self.answer_from_agent(
            question=question,
            agent_result=agent_result,
        )
