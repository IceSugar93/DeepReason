"""paper_search 工具 — 在论文语料（doc_type=paper）中检索相关原文段落。"""

from src.mcp_server.tools.common import compact, safe_run, search

TOOL_SCHEMA = {
    "name": "paper_search",
    "description": (
        "在本地论文库（LLM Agent / RAG 领域的 arXiv 论文）中检索与查询相关的原文段落。"
        "用于核查学术断言是否有论文依据。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询——用断言中的核心术语（英文术语优先），不要整句复制",
            },
            "top_k": {
                "type": "integer",
                "description": "返回段落数，默认 5",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


def paper_search(query: str, top_k: int = 5) -> dict:
    """在论文语料中检索相关段落。"""

    def _run(query: str, top_k: int = 5) -> dict:
        docs = search(query, top_k=top_k, doc_type="paper")
        return {
            "query": query,
            "count": len(docs),
            "results": [compact(d) for d in docs],
        }

    return safe_run(_run, query=query, top_k=top_k)
