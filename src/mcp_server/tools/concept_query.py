"""concept_query 工具 — 跨全语料查询概念/术语的定义与解释。"""

from src.mcp_server.tools.common import compact, safe_run, search

TOOL_SCHEMA = {
    "name": "concept_query",
    "description": (
        "查询某个概念或术语在语料库（论文 + 框架文档 + 博客）中的定义与解释。"
        "适用于核查术语用法、机制定义类断言。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "concept": {
                "type": "string",
                "description": "要查询的概念/术语，如 'Mixture-of-Agents'、'parent-child chunking'",
            },
            "top_k": {
                "type": "integer",
                "description": "返回段落数，默认 3",
                "default": 3,
            },
        },
        "required": ["concept"],
    },
}


def concept_query(concept: str, top_k: int = 3) -> dict:
    """跨全语料查询概念定义。"""

    def _run(concept: str, top_k: int = 3) -> dict:
        docs = search(concept, top_k=top_k, doc_type=None)
        return {
            "concept": concept,
            "count": len(docs),
            "results": [compact(d) for d in docs],
        }

    return safe_run(_run, concept=concept, top_k=top_k)
