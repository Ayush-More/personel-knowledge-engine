from knowledge_engine.agent.gemini_client import GeminiAgentClient
from knowledge_engine.agent.prompts import build_agent_prompt
from knowledge_engine.tool_registry import ToolRegistry
from knowledge_engine.models import RetrievedEvidence , ToolResult
from knowledge_engine.models import AgentRunResult
from knowledge_engine.agent.evidence import deduplicate_evidence
from typing import Any

class AgentRunner:
    def __init__(self , client: GeminiAgentClient , registry: ToolRegistry , max_tool_calls: int = 5)-> None:
        if max_tool_calls <= 0:
            raise ValueError("Max tool call must be greater then zero")

        self.client = client
        self.registry = registry
        self.max_tool_calls = max_tool_calls

    def _build_prompt(self , question: str , history: list[str])-> str:
        base_prompt = build_agent_prompt(self.registry)

        conversation = (
            f"USER QUESTION:\n{question}\n\n"
            f"CONVERSATION HISTORY:\n"
            f"{chr(10).join(history)}"
        )

        return f"{base_prompt}\n\n{conversation}"

    def _execute_tool(self , tool_name: str , argument: dict[str , Any]) -> ToolResult:
        tool = self.registry.get(tool_name)

        return tool.function(**argument)

    def _collect_evidence(self , collected: list[RetrievedEvidence], result:ToolResult) -> None:
        collected.extend(result.evidence)

    
    def run(self, question: str) -> AgentRunResult:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        history: list[str] = []
        collected: list[RetrievedEvidence] = []
        tool_calls_made = 0

        while True:
            prompt = self._build_prompt(question, history)

            decision = self.client.decide(prompt)

            # Gemini has finished gathering evidence.
            if decision.type == "final_answer":
                return AgentRunResult(
                    evidence=deduplicate_evidence(collected),
                    tool_calls_made=tool_calls_made,
                    final_answer=None,
                    status="completed",
                )

            # Never execute more than the allowed number of tools.
            if tool_calls_made >= self.max_tool_calls:
                return AgentRunResult(
                    evidence=deduplicate_evidence(collected),
                    tool_calls_made=tool_calls_made,
                    final_answer=None,
                    status="max_calls_reached",
                )

            tool_call = decision.tool_call

            # Defensive validation, even though Pydantic already validates it.
            if tool_call is None:
                raise ValueError("Tool-call decision contains no tool call")

            # Count every attempt, including failed attempts.
            tool_calls_made += 1

            try:
                result = self._execute_tool(
                    tool_call.tool_name,
                    tool_call.arguments,
                )

                self._collect_evidence(collected, result)

                history.append(
                    f"TOOL RESULT:\n{result.model_dump_json()}"
                )

            except Exception as exc:
                history.append(
                    f"TOOL ERROR:\n"
                    f"Tool: {tool_call.tool_name}\n"
                    f"Error type: {type(exc).__name__}\n"
                    "Choose another available tool or finish if "
                    "sufficient evidence exists."
                )
