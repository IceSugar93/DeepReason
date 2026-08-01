"""MCP 风格工具集 — 供 agent 在进程内调用。

每个工具模块暴露 TOOL_SCHEMA（MCP 风格，call_llm_with_tools 直接消费）
和同名函数本体。对外 MCP server 暴露层属于 Phase 3。
"""

from src.mcp_server.tools.paper_search import TOOL_SCHEMA as PAPER_SEARCH_SCHEMA, paper_search
from src.mcp_server.tools.concept_query import TOOL_SCHEMA as CONCEPT_QUERY_SCHEMA, concept_query
from src.mcp_server.tools.doc_search import TOOL_SCHEMA as FRAMEWORK_DOC_SEARCH_SCHEMA, framework_doc_search
from src.mcp_server.tools.compare import TOOL_SCHEMA as COMPARE_SCHEMA, compare

ALL_TOOL_SCHEMAS = [
    PAPER_SEARCH_SCHEMA,
    CONCEPT_QUERY_SCHEMA,
    FRAMEWORK_DOC_SEARCH_SCHEMA,
    COMPARE_SCHEMA,
]

__all__ = [
    "paper_search",
    "concept_query",
    "framework_doc_search",
    "compare",
    "PAPER_SEARCH_SCHEMA",
    "CONCEPT_QUERY_SCHEMA",
    "FRAMEWORK_DOC_SEARCH_SCHEMA",
    "COMPARE_SCHEMA",
    "ALL_TOOL_SCHEMAS",
]
