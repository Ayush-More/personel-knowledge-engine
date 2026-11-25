import pytest
from unittest.mock import Mock

from knowledge_engine.models import ToolResult
from knowledge_engine.tool_registry import ToolDefinition, ToolRegistry , build_knowledge_tool_registry
from knowledge_engine.models import DocumentChunk
from knowledge_engine.tools import KnowledgeTools


def dummy_tool(query: str) -> ToolResult:
    return ToolResult(
        tool_name="dummy_tool",
        query=query,
        evidence=[],
        result_count=0,
    )


def test_registry_registers_and_retrieves_tool():
    registry = ToolRegistry()

    definition = ToolDefinition(
        name="dummy_tool",
        description="A test tool",
        function=dummy_tool,
    )

    registry.register(definition)

    assert registry.get("dummy_tool") is definition


def test_registry_rejects_duplicate_tool_names():
    registry = ToolRegistry()

    definition = ToolDefinition(
        name="dummy_tool",
        description="A test tool",
        function=dummy_tool,
    )

    registry.register(definition)

    with pytest.raises(ValueError, match="Tool already registered"):
        registry.register(definition)


def test_registry_rejects_unknown_tool():
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="Unknown tool"):
            registry.get("missing_tool")


def test_registry_lists_registered_tools():
        registry = ToolRegistry()

        definition = ToolDefinition(
            name="dummy_tool",
            description="A test tool",
            function=dummy_tool,
        )

        registry.register(definition)

        assert registry.list_tools() == [definition]


def test_registry_contains_real_knowledge_tools():
    pipeline = Mock()
    knowledge_tools = KnowledgeTools(pipeline)

    registry = build_knowledge_tool_registry(knowledge_tools)

    assert registry.get("search_projects").name == "search_projects"
    assert registry.get("search_skills").name == "search_skills"

    assert len(registry.list_tools()) == 2



def test_registered_project_tool_executes_successfully():
    chunk = DocumentChunk(
        chunk_id="project-1",
        document_id="doc-1",
        content="Built a Python application",
        chunk_index=0,
        start_char=0,
        end_char=28,
        source="data/project-campusconnect.md",
        metadata={"category": "projects"},
    )

    pipeline = Mock()
    pipeline.search.return_value = [(chunk, 0.85)]

    knowledge_tools = KnowledgeTools(pipeline)
    registry = build_knowledge_tool_registry(knowledge_tools)

    tool = registry.get("search_projects")

    result = tool.function(
        query="Python projects",
        top_k=5,
    )

    assert result.tool_name == "search_projects"
    assert result.result_count == 1
    assert result.evidence[0].chunk_id == "project-1"

    pipeline.search.assert_called_once_with(
        query="Python projects",
        top_k=5,
        category="projects",
    )


def test_registered_skills_tool_executes_successfully():

    pipeline = Mock()
    pipeline.search.return_value = []

    knowledge_tools = KnowledgeTools(pipeline)
    registry = build_knowledge_tool_registry(knowledge_tools)

    tool = registry.get("search_skills")

    result = tool.function(
        query="Python and machine learning",
        top_k=3,
    )

    assert result.tool_name == "search_skills"
    assert result.query == "Python and machine learning"

    pipeline.search.assert_called_once_with(
        query="Python and machine learning",
        top_k=3,
        category="skills",
    )
