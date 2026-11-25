from dataclasses import dataclass
from collections.abc import Callable
from knowledge_engine.tools import KnowledgeTools

from knowledge_engine.models import ToolResult

@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    function: Callable[...,ToolResult]

class ToolRegistry:
    """Register and discover tools available to the agent."""

    def __init__(self)-> None:
        self._tools: dict[str , ToolDefinition] = {}

    def register(self , tool: ToolDefinition)-> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self , name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")

        return self._tools[name]

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

def build_knowledge_tool_registry(knowledge_tools: KnowledgeTools) -> ToolRegistry:
    """Build a registry containing the available knowledge tools."""

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="search_projects",
            description=(
                "Search project-related knowledge , including projects"
                "details , technologies , responsibilities and achivements."
            ),
            function=knowledge_tools.search_projects
        )
    )

    registry.register(
        ToolDefinition(
            name="search_skills",
            description=(
                "Search knowledge about technical skills , programming"
                "languages , frameworks and other competencies"
            ),
            function= knowledge_tools.search_skills
        )
    )

    return registry