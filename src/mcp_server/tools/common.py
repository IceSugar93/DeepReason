"""工具共享助手 — 检索后端访问、结果压缩与执行兜底。"""

from config.settings import FINAL_TOP_K
from src.mcp_server.runtime import get_retriever

VALID_DOC_TYPES = {"paper", "doc", "blog"}

# snippet 截断长度：tool 结果要进 LLM 上下文，单条不宜过长
SNIPPET_MAX_CHARS = 600


def search(query: str, top_k: int = FINAL_TOP_K, doc_type: str | None = None) -> list[dict]:
    """经注册的 HybridRetriever 执行检索。

    跳过 HyDE——工具查询已是具体断言/术语文本，每次一次 LLM 假设答案
    生成的开销不值得。后端未注册时抛 RuntimeError（由 safe_run 兜底）。
    """
    retriever = get_retriever()
    if retriever is None:
        raise RuntimeError("retriever backend not registered")

    # 复用 hybrid_retriever 的模块级 embedding 缓存，避免重复加载 BGE-M3
    from src.retrieval.hybrid_retriever import _get_embedding

    return retriever.search(
        query=query,
        query_embedding=_get_embedding(query),
        top_k=top_k,
        doc_type_filter=doc_type,
        skip_hyde=True,
    )


def compact(doc: dict, max_chars: int = SNIPPET_MAX_CHARS) -> dict:
    """把检索文档压缩为适合进 LLM 上下文的精简 JSON。

    paper 类型的标题/作者/发表日期从 parent 缓存补齐（检索 hit 的 title
    字段对论文而言是 PDF 页眉噪声，不可用）。
    """
    item = {
        "chunk_id": doc.get("chunk_id", ""),
        "title": doc.get("title", ""),
        "doc_type": doc.get("doc_type", ""),
        "snippet": doc.get("content", "")[:max_chars],
        "score": round(float(doc.get("score", 0)), 4),
    }
    if doc.get("doc_type") == "paper":
        from src.retrieval.vector_store import load_parents

        parent = load_parents().get(doc.get("chunk_id", ""), {})
        item["title"] = parent.get("paper_title") or item["title"]
        item["authors"] = parent.get("paper_authors", "")
        item["published"] = parent.get("paper_published", "")
    return item


def safe_run(fn, **kwargs) -> dict:
    """工具执行兜底：任何异常都返回 error dict，而不是抛给正在调工具的 LLM。"""
    try:
        return fn(**kwargs)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
