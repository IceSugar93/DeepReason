"""文档格式化工具 — 将检索文档列表格式化为带编号的文献文本。

原先放在 src/agents/advocate.py，3-Agent 架构重构后抽取为共享工具。
"""


def format_docs(docs: list[dict]) -> str:
    """将检索文档列表格式化为带编号的文献文本。

    Args:
        docs: 检索到的文档列表（含 title, chunk_id, content 等字段）。

    Returns:
        格式化的文献文本，每篇以 "[来源 N] title (ID: xxx)" 开头，
        后紧跟 content。
    """
    parts = []
    for i, doc in enumerate(docs):
        title = doc.get("title", "未知来源")
        chunk_id = doc.get("chunk_id", f"doc_{i}")
        content = doc.get("content", "")
        parts.append(f"[来源 {i+1}] {title} (ID: {chunk_id})\n{content}\n")
    return "\n".join(parts) if parts else "（无可用的参考文献）"
