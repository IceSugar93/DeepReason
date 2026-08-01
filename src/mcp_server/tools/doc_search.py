"""framework_doc_search 工具 — 在框架官方文档语料（doc_type=doc）中检索。

语料范围：LangChain / LangGraph / MCP 等框架的官方文档。
"""

from src.mcp_server.tools.common import compact, safe_run, search

TOOL_SCHEMA = {
    "name": "framework_doc_search",
    "description": (
        "在框架官方文档（LangChain / LangGraph / MCP 等）中检索与查询相关的段落。"
        "适用于核查框架 API、架构机制、协议规范类断言。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询——用框架名 + 具体机制/API 名",
            },
            "top_k": {
                "type": "integer",
                "description": "返回段落数，默认 3",
                "default": 3,
            },
        },
        "required": ["query"],
    },
}


def framework_doc_search(query: str, top_k: int = 3) -> dict:
    """在框架文档语料中检索。"""

    def _run(query: str, top_k: int = 3) -> dict:
        docs = search(query, top_k=top_k, doc_type="doc")
        return {
            "query": query,
            "count": len(docs),
            "results": [compact(d) for d in docs],
        }

    return safe_run(_run, query=query, top_k=top_k)
