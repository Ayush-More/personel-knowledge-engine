
from unittest.mock import Mock

import pytest

from knowledge_engine.agent.runner import AgentRunner
from knowledge_engine.models import (
    AgentDecision,
    AgentRunResult,
    RetrievedEvidence,
    ToolCall,
    ToolResult,
)
from knowledge_engine.tool_registry import ToolDefinition, ToolRegistry


def make_evidence(
    chunk_id: str = "chunk-1",
    score: float = 0.85,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id="doc-1",
        content="Python experience",
        chunk_index=0,
        start_char=0,
        end_char=18,
        source="experience.md",
        metadata={"category": "experience"},
        score=score,
    )


def make_tool_decision(
    tool_name: str = "search_projects",
    arguments: dict | None = None,
) -> AgentDecision:
    return AgentDecision(
        type="tool_call",
        tool_call=ToolCall(
            tool_name=tool_name,
            arguments=arguments or {"query": "Python"},
        ),
    )


def make_final_decision() -> AgentDecision:
    return AgentDecision(
        type="final_answer",
        answer="Evidence gathering is complete.",
    )


def make_tool_result(
    evidence: list[RetrievedEvidence],
    tool_name: str = "search_projects",
) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        query="Python",
        evidence=evidence,
        result_count=len(evidence),
    )



def test_run_executes_tool_then_returns_completed_result():
    evidence = make_evidence()

    client = Mock()
    client.decide.side_effect = [
        make_tool_decision(),
        make_final_decision(),
    ]

    registry = ToolRegistry()

    mock_tool = Mock(
        return_value=make_tool_result([evidence])
    )

    registry.register(
        ToolDefinition(
            name="search_projects",
            description="Search projects",
            function=mock_tool,
        )
    )

    runner = AgentRunner(client, registry)

    result = runner.run("What projects use Python?")

    assert isinstance(result, AgentRunResult)
    assert result.status == "completed"
    assert result.tool_calls_made == 1
    assert result.evidence == [evidence]

    mock_tool.assert_called_once_with(query="Python")
    assert client.decide.call_count == 2


def test_run_collects_evidence_from_multiple_tool_calls():
    project_evidence = make_evidence("project-1")
    skill_evidence = make_evidence("skill-1")

    client = Mock()
    client.decide.side_effect = [
        make_tool_decision("search_projects"),
        make_tool_decision(
            "search_skills",
            {"query": "Python"},
        ),
        make_final_decision(),
    ]

    registry = ToolRegistry()

    project_tool = Mock(
        return_value=make_tool_result(
            [project_evidence],
            "search_projects",
        )
    )

    skill_tool = Mock(
        return_value=make_tool_result(
            [skill_evidence],
            "search_skills",
        )
    )

    registry.register(
        ToolDefinition(
            name="search_projects",
            description="Search projects",
            function=project_tool,
        )
    )

    registry.register(
        ToolDefinition(
            name="search_skills",
            description="Search skills",
            function=skill_tool,
        )
    )

    runner = AgentRunner(client, registry)

    result = runner.run("Tell me about my Python experience")

    assert result.status == "completed"
    assert result.tool_calls_made == 2
    assert len(result.evidence) == 2

    assert project_tool.call_count == 1
    assert skill_tool.call_count == 1
    assert client.decide.call_count == 3



def test_run_deduplicates_evidence_and_keeps_highest_score():
    first = make_evidence("chunk-1", 0.70)
    second = make_evidence("chunk-1", 0.95)

    client = Mock()
    client.decide.side_effect = [
        make_tool_decision("search_projects"),
        make_tool_decision("search_skills"),
        make_final_decision(),
    ]

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="search_projects",
            description="Search projects",
            function=Mock(
                return_value=make_tool_result([first])
            ),
        )
    )

    registry.register(
        ToolDefinition(
            name="search_skills",
            description="Search skills",
            function=Mock(
                return_value=make_tool_result(
                    [second],
                    "search_skills",
                )
            ),
        )
    )

    runner = AgentRunner(client, registry)

    result = runner.run("Tell me about Python")

    assert result.status == "completed"
    assert len(result.evidence) == 1
    assert result.evidence[0].chunk_id == "chunk-1"
    assert result.evidence[0].score == 0.95



def test_run_continues_after_tool_execution_failure():
    client = Mock()
    client.decide.side_effect = [
        make_tool_decision(),
        make_final_decision(),
    ]

    registry = ToolRegistry()

    failing_tool = Mock(
        side_effect=RuntimeError("Internal sensitive details")
    )

    registry.register(
        ToolDefinition(
            name="search_projects",
            description="Search projects",
            function=failing_tool,
        )
    )

    runner = AgentRunner(client, registry)

    result = runner.run("Find my projects")

    assert result.status == "completed"
    assert result.tool_calls_made == 1
    assert result.evidence == []

    assert client.decide.call_count == 2

    second_prompt = client.decide.call_args_list[1].args[0]

    assert "TOOL ERROR" in second_prompt
    assert "RuntimeError" in second_prompt
    assert "Internal sensitive details" not in second_prompt



def test_run_never_executes_more_than_max_tool_calls():
    client = Mock()

    # Gemini keeps requesting tools.
    client.decide.side_effect = [
        make_tool_decision(),
        make_tool_decision(),
        make_tool_decision(),
        make_tool_decision(),
        make_tool_decision(),
        make_tool_decision(),
    ]

    registry = ToolRegistry()

    mock_tool = Mock(
        return_value=make_tool_result([make_evidence()])
    )

    registry.register(
        ToolDefinition(
            name="search_projects",
            description="Search projects",
            function=mock_tool,
        )
    )

    runner = AgentRunner(
        client,
        registry,
        max_tool_calls=5,
    )

    result = runner.run("Find my projects")

    assert result.status == "max_calls_reached"
    assert result.tool_calls_made == 5

    # Five executions, never six.
    assert mock_tool.call_count == 5

    # Six decisions: five tool calls and one final
    # opportunity to decide whether to finish.
    assert client.decide.call_count == 6



def test_run_rejects_empty_question():
    client = Mock()
    registry = ToolRegistry()

    runner = AgentRunner(client, registry)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        runner.run("   ")

    client.decide.assert_not_called()
