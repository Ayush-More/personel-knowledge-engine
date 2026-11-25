
import pytest

from knowledge_engine.answering import KnowledgeAnswer
from knowledge_engine.models import (
    AgentRunResult,
    GeneratedAnswer,
    RetrievedEvidence,
)


class FakeGenerator:
    def __init__(self):
        self.received_question = None
        self.received_results = None

    def generate(self, question, result):
        self.received_question = question
        self.received_results = result

        return GeneratedAnswer(
            question=question,
            answer="Test grounded answer",
            citations=[],
            insufficient_evidence=False,
        )


def make_evidence():
    return RetrievedEvidence(
        chunk_id="chunk-001",
        document_id="doc-001",
        content="I built a personal knowledge engine.",
        chunk_index=0,
        start_char=0,
        end_char=39,
        source="projects.md",
        metadata={"category": "projects"},
        score=0.92,
    )


def test_answer_from_agent_converts_evidence():
    generator = FakeGenerator()
    answerer = KnowledgeAnswer(pipeline=None, generator=generator)

    agent_result = AgentRunResult(
        evidence=[make_evidence()],
        tool_calls_made=1,
        status="completed",
    )

    answerer.answer_from_agent("What did I build?", agent_result)

    received_chunk, received_score = generator.received_results[0]

    assert received_chunk.chunk_id == "chunk-001"
    assert received_chunk.document_id == "doc-001"
    assert received_chunk.content == "I built a personal knowledge engine."
    assert received_chunk.source == "projects.md"
    assert received_chunk.metadata == {"category": "projects"}
    assert received_score == 0.92


def test_answer_from_agent_passes_empty_evidence():
    generator = FakeGenerator()
    answerer = KnowledgeAnswer(pipeline=None, generator=generator)

    agent_result = AgentRunResult(
        evidence=[],
        tool_calls_made=0,
        status="completed",
    )

    answerer.answer_from_agent("What did I build?", agent_result)

    assert generator.received_results == []


def test_answer_from_agent_rejects_failed_run():
    generator = FakeGenerator()
    answerer = KnowledgeAnswer(pipeline=None, generator=generator)

    agent_result = AgentRunResult(
        evidence=[make_evidence()],
        tool_calls_made=1,
        status="failed",
    )

    with pytest.raises(RuntimeError):
        answerer.answer_from_agent("What did I build?", agent_result)

    assert generator.received_results is None


def test_answer_from_agent_rejects_empty_question():
    generator = FakeGenerator()
    answerer = KnowledgeAnswer(pipeline=None, generator=generator)

    agent_result = AgentRunResult(
        evidence=[make_evidence()],
        tool_calls_made=1,
        status="completed",
    )

    with pytest.raises(ValueError):
        answerer.answer_from_agent("   ", agent_result)


class FakeAgentRunner:
    def __init__(self, result):
        self.result = result
        self.received_question = None

    def run(self, question):
        self.received_question = question
        return self.result


def test_answer_with_agent_runs_agent_and_generates_answer():
    generator = FakeGenerator()

    agent_result = AgentRunResult(
        evidence=[make_evidence()],
        tool_calls_made=1,
        status="completed",
    )

    agent_runner = FakeAgentRunner(agent_result)

    answerer = KnowledgeAnswer(
        pipeline=None,
        generator=generator,
        agent_runner=agent_runner,
    )

    answer = answerer.answer_with_agent("What did I build?")

    assert agent_runner.received_question == "What did I build?"
    assert generator.received_question == "What did I build?"
    assert generator.received_results[0][0].chunk_id == "chunk-001"
    assert answer.answer == "Test grounded answer"


def test_answer_with_agent_rejects_missing_agent_runner():
    answerer = KnowledgeAnswer(
        pipeline=None,
        generator=FakeGenerator(),
    )

    with pytest.raises(RuntimeError, match="AgentRunner is not configured"):
        answerer.answer_with_agent("What did I build?")


def test_answer_with_agent_rejects_failed_agent_run():
    generator = FakeGenerator()

    agent_result = AgentRunResult(
        evidence=[],
        tool_calls_made=1,
        status="failed",
    )

    agent_runner = FakeAgentRunner(agent_result)

    answerer = KnowledgeAnswer(
        pipeline=None,
        generator=generator,
        agent_runner=agent_runner,
    )

    with pytest.raises(RuntimeError):
        answerer.answer_with_agent("What did I build?")

    assert generator.received_results is None

