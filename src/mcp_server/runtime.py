"""工具后端注册表 — 进程内共享 HybridRetriever 实例。

引擎（DeepReasonEngine.run）在图执行前经 set_retriever_context 注册检索器；
mcp_server 下的工具函数在图内被 agent 调用时，通过 get_retriever() 取回
同一实例。进程内调用不走 MCP 协议——对外 MCP server 暴露层属于 Phase 3。
"""

_registry: dict = {}


def register_retriever(retriever) -> None:
    """注册工具共享的检索器后端（由引擎在图执行前调用）。"""
    _registry["retriever"] = retriever


def get_retriever():
    """取回已注册的检索器；未注册时返回 None（工具应返回 error dict）。"""
    return _registry.get("retriever")
