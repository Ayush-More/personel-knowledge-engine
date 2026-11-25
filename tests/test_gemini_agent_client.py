from unittest.mock import MagicMock
import pytest
from knowledge_engine.agent.gemini_client import GeminiAgentClient
from knowledge_engine.models import AgentDecision, ToolCall


def test_decide_returns_structured_tool_call():
    client = GeminiAgentClient.__new__(GeminiAgentClient)

    expected = AgentDecision(
        type="tool_call",
        tool_call=ToolCall(
            tool_name="search_projects",
            arguments={"query": "Python"},
        ),
    )

    mock_response = MagicMock()
    mock_response.parsed = expected

    client.client = MagicMock()
    client.client.models.generate_content.return_value = mock_response
    client.model = "test-model"

    result = client.decide("Find projects using Python")

    assert isinstance(result, AgentDecision)
    assert result.type == "tool_call"
    assert result.tool_call.tool_name == "search_projects"


def test_decide_returns_final_answer():
    client = GeminiAgentClient.__new__(GeminiAgentClient)

    expected = AgentDecision(
        type="final_answer",
        answer="I found two projects using Python.",
    )

    mock_response = MagicMock()
    mock_response.parsed = expected

    client.client = MagicMock()
    client.client.models.generate_content.return_value = mock_response
    client.model = "test-model"

    result = client.decide("Which projects used Python?")

    assert isinstance(result, AgentDecision)
    assert result.type == "final_answer"
    assert result.answer == "I found two projects using Python."

def test_decide_raises_error_when_response_is_missing():
    client = GeminiAgentClient.__new__(GeminiAgentClient)

    mock_response = MagicMock()
    mock_response.parsed = None

    client.client = MagicMock()
    client.client.models.generate_content.return_value = mock_response
    client.model = "test-model"

    with pytest.raises(ValueError, match="Gemini returned no structured decision"):
        client.decide("Find my projects")

def test_decide_rejects_empty_prompt():
    client = GeminiAgentClient.__new__(GeminiAgentClient)

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.decide("   ")