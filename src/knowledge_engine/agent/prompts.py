from knowledge_engine.tool_registry import ToolRegistry

def format_tool_descriptions(registry: ToolRegistry)-> str:

    tools = registry.list_tools()

    descriptions = []

    for tool in tools:
        descriptions.append(
            f"Tool name: {tool.name}\n"
            f"Description: {tool.description}"
        )

    return "\n\n".join(descriptions)

def build_agent_prompt(registry: ToolRegistry) -> str:
    tool_descriptions = format_tool_descriptions(registry)

    return(
        f"{AGENT_SYSTEM_PROMPT}\n\n"
        f"AVAILABLE TOOLS:\n"
        f"{tool_descriptions}\n\n"
        "You must choose an available tool when more knowledge "
        "is needed. Do not invent tool names."
    )


AGENT_SYSTEM_PROMPT = """
You are a personal knowledge assistant.

Your responsibility is to decide which available knowledge tools
should be used to retrieve information for the user's question

You must:
1. Use only the tools provided to you
2. Choose tool based on the information needed
3. Never invent retrieved evidence.
4. Do not assume information exist if tool returns no evidence.
5. Stop requesting tools when sufficient evidence has been collected.

You are responsible for tool selection and orchestration,
not final answer generation
"""