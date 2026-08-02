"""统一 LLM 调用封装 — 直接使用 openai 库调用 DeepSeek API

DeepSeek 提供 OpenAI 兼容的 API，直接通过 `openai` 包调用，
无需依赖 langchain。后续如需接入 Claude 等模型可再扩展。
"""

import json

from openai import OpenAI

from config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    PLANNER_MODEL, GENERATOR_MODEL, REVISER_MODEL,
    CRITIC_MODEL,
    AGENT_TIMEOUT,
)

# 重新导出模型名，方便其他模块统一导入
__all__ = [
    "call_llm", "call_llm_with_json", "call_llm_with_tools", "call_llm_tool_loop",
    "execute_tool_call",
    "PLANNER_MODEL", "GENERATOR_MODEL", "REVISER_MODEL",
    "CRITIC_MODEL",
]

# 延迟初始化的共享客户端（连接复用，性能更好）
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """返回指向 DeepSeek 端点的单例 OpenAI 客户端。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            # Guardrails: 单次请求超时兜底，网络/网关卡死时抛超时而非挂起
            timeout=AGENT_TIMEOUT,
        )
    return _client


def call_llm(model_name: str, system_prompt: str, user_prompt: str,
             temperature: float = 0.3) -> str:
    """单次 LLM 调用，返回纯文本。

    Args:
        model_name: 模型标识，如 "deepseek-chat"。
        system_prompt: 系统角色提示词。
        user_prompt: 用户消息。
        temperature: 采样温度。

    Returns:
        LLM 响应文本。
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
                       temperature: float = 0.1, return_raw: bool = False):
    """LLM 调用，强制返回 JSON 格式（通过 DeepSeek JSON Mode）。

    注意：DeepSeek JSON Mode 要求 system 或 user prompt 中出现 "json" 字样，
    编写 prompt 时请确保满足此条件。

    Args:
        model_name: 模型标识。
        system_prompt: 系统角色提示词（应提及 JSON 输出）。
        user_prompt: 用户消息。
        temperature: 结构化输出建议使用低温度。
        return_raw: 为 True 时返回 {"parsed": dict, "raw": str}，
            保留模型输出的原始文本（调试/完整展示用）。

    Returns:
        解析后的 JSON dict；return_raw=True 时返回含 parsed/raw 的 dict。
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
    parsed = json.loads(content)
    if return_raw:
        return {"parsed": parsed, "raw": content}
    return parsed


def _to_openai_tool_format(mcp_tool_schema: dict) -> dict:
    """将 MCP 风格的 tool schema 转换为 OpenAI function-calling 格式。

    MCP schema:  {"name": ..., "description": ..., "inputSchema": {...}}
    OpenAI 期望: {"type": "function",
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
    """LLM 调用，支持 Function Calling / Tool Use。

    Args:
        model_name: 模型标识。
        system_prompt: 系统角色提示词。
        user_prompt: 用户消息。
        tools: MCP 风格的 tool schema 列表（内部自动转换为 OpenAI 格式）。
        temperature: 采样温度。

    Returns:
        {"type": "tool_calls", "tool_calls": [{"name", "args"}]}
        或 {"type": "text", "content": <str>}
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
    """将 tool call 路由到对应的 MCP 工具函数并执行。

    Args:
        tool_name: 要调用的工具名称。
        tool_args: 传给工具的参数 dict。

    Returns:
        工具执行结果，序列化为 JSON 字符串。
    """
    # 延迟导入以避免模块加载时的循环引用
    # （工具模块本身可能从 src.utils 导入）
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
        return json.dumps({"error": f"未知工具: {tool_name}"})

    result = func(**tool_args)
    return json.dumps(result, ensure_ascii=False)


def call_llm_tool_loop(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
    executor=execute_tool_call,
    max_rounds: int = 3,
    temperature: float = 0.1,
) -> dict:
    """多轮工具调用循环：LLM 可连续调用工具，直到给出最终文本回答。

    与 call_llm_with_tools（单轮）的区别：内部维护 messages 累积，
    每轮把 assistant 的 tool_calls 和工具回执（role=tool）追加进上下文，
    直到 LLM 不再发起工具调用或达到轮数上限。

    Args:
        model_name: 模型标识。
        system_prompt: 系统角色提示词。
        user_prompt: 用户消息。
        tools: MCP 风格的 tool schema 列表。
        executor: 执行单个工具调用的函数 (name, args) -> JSON 字符串。
        max_rounds: 最大工具调用轮数，超出后返回 truncated=True。
        temperature: 采样温度。

    Returns:
        {"text": 最终文本, "tool_log": [{"name", "args", "result"}], "truncated": bool}
    """
    client = _get_client()
    openai_tools = [_to_openai_tool_format(t) for t in tools]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tool_log: list[dict] = []

    message = None
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=openai_tools,
            temperature=temperature,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return {"text": message.content or "", "tool_log": tool_log, "truncated": False}

        messages.append(message)
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result_str = executor(tc.function.name, args)
            tool_log.append({"name": tc.function.name, "args": args, "result": result_str})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # 轮数耗尽但工具结果已在上下文中：强制模型基于已检索材料终答。
    # 注意不能简单去掉 tools 参数——deepseek-v4 在工具调用轨迹中段被切断时
    # 会把 DSML 工具标记当普通文本输出。正确做法是保留 tools 并用
    # tool_choice="none" 显式禁止继续调用，同时追加一条收尾指令。
    messages.append({
        "role": "user",
        "content": "检索轮次已用完。禁止再调用任何工具，请立即基于已检索到的材料给出最终答案。",
    })
    final_text = ""
    try:
        final = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=openai_tools,
            tool_choice="none",
            temperature=temperature,
        )
        final_text = final.choices[0].message.content or ""
    except Exception:
        # 网关不支持 tool_choice="none" 时退化：不带 tools 重试一次
        try:
            final = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
            final_text = final.choices[0].message.content or ""
        except Exception:
            final_text = ""

    return {
        "text": final_text,
        "tool_log": tool_log,
        "truncated": True,
    }
