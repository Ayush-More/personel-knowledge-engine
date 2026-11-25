from knowledge_engine.agent.prompts import format_tool_descriptions , build_agent_prompt

from knowledge_engine.agent.prompts import format_tool_descriptions
from knowledge_engine.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)


def test_format_tool_descriptions():
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="search_projects",
            description="Search project information.",
            function=lambda **kwargs: None,
        )
    )

    registry.register(
        ToolDefinition(
            name="search_skills",
            description="Search technical skills.",
            function=lambda **kwargs: None,
        )
    )

    result = format_tool_descriptions(registry)

    assert "search_projects" in result
    assert "Search project information." in result
    assert "search_skills" in result
    assert "Search technical skills." in result

def test_build_agent_prompt_includes_system_prompt_and_tools():
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="search_projects",
            description="Search project information.",
            function=lambda **kwargs: None,
        )
    )

    prompt = build_agent_prompt(registry)

    assert "personal knowledge assistant" in prompt
    assert "search_projects" in prompt
    assert "Search project information." in prompt
    assert "Do not invent tool names." in prompt