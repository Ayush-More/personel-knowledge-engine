import pytest
from pydantic import ValidationError

from knowledge_engine.models import AgentDecision, ToolCall


def test_tool_call_decision_is_valid():
    decision = AgentDecision(
        type="tool_call",
        tool_call=ToolCall(
            tool_name="search_projects",
            arguments={"query": "Python", "top_k": 5},
        ),
    )

    assert decision.type == "tool_call"
    assert decision.tool_call.tool_name == "search_projects"


def test_final_answer_decision_is_valid():
    decision = AgentDecision(
        type="final_answer",
        answer="I found two projects.",
    )

    assert decision.type == "final_answer"
    assert decision.answer == "I found two projects."


def test_tool_call_cannot_contain_final_answer():
    with pytest.raises(ValidationError):
        AgentDecision(
            type="tool_call",
            answer="This should not be allowed",
        )


def test_final_answer_cannot_contain_tool_call():
    with pytest.raises(ValidationError):
        AgentDecision(
            type="final_answer",
            answer="Here is the answer",
            tool_call=ToolCall(tool_name="search_projects"),
        )


def test_final_answer_cannot_be_empty():
    with pytest.raises(ValidationError):
        AgentDecision(
            type="final_answer",
            answer="   ",
        )