"""Unified LLM calling wrapper using the openai library directly.

DeepSeek uses an OpenAI-compatible API, so we can call it via the standard
`openai` package without depending on langchain. The Anthropic (Claude) SDK
can be added later if needed.
"""

import json

from openai import OpenAI

from config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    PLANNER_MODEL, ADVOCATE_MODEL, SKEPTIC_MODEL,
    JUDGE_MODEL, VALIDATOR_MODEL,
)

# Re-export model names so other modules can import them from here if convenient.
__all__ = [
    "call_llm", "call_llm_with_json", "call_llm_with_tools",
    "execute_tool_call",
    "PLANNER_MODEL", "ADVOCATE_MODEL", "SKEPTIC_MODEL",
    "JUDGE_MODEL", "VALIDATOR_MODEL",
]

# Lazily-initialized shared client (connection pooling, better performance).
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return a singleton OpenAI client pointing at the DeepSeek endpoint."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    return _client


def call_llm(model_name: str, system_prompt: str, user_prompt: str,
             temperature: float = 0.3) -> str:
    """Single LLM call returning plain text.

    Args:
        model_name: Model identifier, e.g. "deepseek-chat".
        system_prompt: System role prompt.
        user_prompt: User message.
        temperature: Sampling temperature.

    Returns:
        LLM response text.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def call_llm_with_json(model_name: str, system_prompt: str, user_prompt: str,
                       temperature: float = 0.1) -> dict:
    """LLM call forcing JSON output via DeepSeek's JSON mode.

    DeepSeek requires the word "json" to appear somewhere in the system or
    user prompt when using JSON mode — make sure your prompts satisfy this.

    Args:
        model_name: Model identifier.
        system_prompt: System role prompt (should mention JSON output).
        user_prompt: User message.
        temperature: Low temperature recommended for structured output.

    Returns:
        Parsed JSON dict.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return json.loads(content)


def _to_openai_tool_format(mcp_tool_schema: dict) -> dict:
    """Convert an MCP-style tool schema to the OpenAI function-calling format.

    MCP schema: {"name": ..., "description": ..., "inputSchema": {...}}
    OpenAI tools expect: {"type": "function",
                          "function": {"name", "description", "parameters"}}
    """
    return {
        "type": "function",
        "function": {
            "name": mcp_tool_schema["name"],
            "description": mcp_tool_schema["description"],
            "parameters": mcp_tool_schema["inputSchema"],
        },
    }


def call_llm_with_tools(model_name: str, system_prompt: str, user_prompt: str,
                        tools: list[dict], temperature: float = 0.3) -> dict:
    """LLM call with Function Calling / Tool Use support.

    Args:
        model_name: Model identifier.
        system_prompt: System role prompt.
        user_prompt: User message.
        tools: List of MCP-style tool schemas (will be converted internally).
        temperature: Sampling temperature.

    Returns:
        {"type": "tool_calls", "tool_calls": [{"name", "args"}]}
        or {"type": "text", "content": <str>}
    """
    client = _get_client()
    openai_tools = [_to_openai_tool_format(t) for t in tools]

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=openai_tools,
        temperature=temperature,
    )
    message = response.choices[0].message

    if message.tool_calls:
        return {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ],
        }
    return {"type": "text", "content": message.content}


def execute_tool_call(tool_name: str, tool_args: dict) -> str:
    """Route a tool call to the matching MCP tool function.

    Args:
        tool_name: Name of the tool to invoke.
        tool_args: Arguments dict for the tool.

    Returns:
        Tool execution result serialized as a JSON string.
    """
    # Imported here to avoid a circular import at module load time
    # (the tools themselves may import from src.utils).
    from src.mcp_server.tools.paper_search import paper_search
    from src.mcp_server.tools.concept_query import concept_query
    from src.mcp_server.tools.doc_search import framework_doc_search
    from src.mcp_server.tools.compare import compare

    tool_map = {
        "paper_search": paper_search,
        "concept_query": concept_query,
        "framework_doc_search": framework_doc_search,
        "compare": compare,
    }

    func = tool_map.get(tool_name)
    if func is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    result = func(**tool_args)
    return json.dumps(result, ensure_ascii=False)
