"""compare 工具 — 对两个实体分别检索，返回并排结果供对比分析。"""

from src.mcp_server.tools.common import compact, safe_run, search

TOOL_SCHEMA = {
    "name": "compare",
    "description": (
        "对两个实体（论文方法 / 框架 / 概念）分别检索相关段落，返回两组并排结果，"
        "供对比分析异同点。适用于对比类问题。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "entity_a": {"type": "string", "description": "第一个实体名称"},
            "entity_b": {"type": "string", "description": "第二个实体名称"},
            "top_k": {
                "type": "integer",
                "description": "每个实体返回段落数，默认 4",
                "default": 4,
            },
        },
        "required": ["entity_a", "entity_b"],
    },
}


def compare(entity_a: str, entity_b: str, top_k: int = 4) -> dict:
    """对两个实体分别检索并返回并排结果。"""

    def _run(entity_a: str, entity_b: str, top_k: int = 4) -> dict:
        docs_a = search(entity_a, top_k=top_k)
        docs_b = search(entity_b, top_k=top_k)
        return {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "results_a": [compact(d) for d in docs_a],
            "results_b": [compact(d) for d in docs_b],
        }

    return safe_run(_run, entity_a=entity_a, entity_b=entity_b, top_k=top_k)
