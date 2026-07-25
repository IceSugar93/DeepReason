"""LangGraph 状态定义 — DeepReason 3-Agent 审查-修订推理引擎

AgentState 是图的所有节点共享的 TypedDict，包含检索结果、审查状态、
修订历史和最终输出。使用 Annotated 累加器处理追加型字段。

架构简化（2026-07-24）：从 5 Agent（Planner/Advocate/Skeptic/Judge/Validator）
精简为 3 Agent（Generator/Critic/Reviser）+ 检索链。
"""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """3-Agent 审查-修订推理的全局状态。

    Generator → Critic → Reviser → Critic 循环（必要时），accept 时输出。
    带 Annotated[list, operator.add] 的字段为累加器——节点只需返回
    ``{"field": [item]}`` 即可自动追加。
    """

    # ── Input ──────────────────────────────────────────────────────────
    query: str

    # ── Retrieval ───────────────────────────────────────────────────────
    retrieved_docs: list[dict]
    sub_question_docs: dict[str, list[dict]]

    # ── Planning ────────────────────────────────────────────────────────
    complexity: str                         # "simple" | "multi_hop"
    sub_questions: list[str]

    # ── Generation ──────────────────────────────────────────────────────
    draft_answer: str                       # Generator 初稿

    # ── Review Loop ─────────────────────────────────────────────────────
    review_round: int                       # 当前审查轮次 (1..max)
    critic_ruling: dict                     # {"verdict", "confidence", "issues", "improvements", "reasoning"}
    review_history: Annotated[list[dict], operator.add]

    # ── Revision ────────────────────────────────────────────────────────
    revision_round: int                     # 当前修订轮次 (0..max)
    refined_answer: str                     # Reviser 最新修订版本
    previous_issues: list[dict]             # 上一轮 Critic(full) 中 conf>0.8 的 issues，供 Reviser 定向修改和 Verifier 定向检查
    revision_history: Annotated[list[dict], operator.add]

    # ── Verification ────────────────────────────────────────────────────
    verify_result: dict                     # {"all_resolved": bool, "details": [{issue_index, resolved, reason}]}

    # ── Output ─────────────────────────────────────────────────────────
    final_answer: str
    confidence: float

    # ── Control ────────────────────────────────────────────────────────
    retrieval_hops: int
    errors: Annotated[list[str], operator.add]
