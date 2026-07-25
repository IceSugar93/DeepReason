"""图构建器 — 3-Agent 审查-修订引擎（含 Verify 定向检查）

生成：
  retrieve → plan ──→ multi_hop_retrieve ──→ generator
                    └─────────────────────→ generator
    → critic(full)
        → accept → finalize
        → 有高置信度 issue → reviser → verify
            → 全部解决 → critic(full)  ← 重新全面审查
            → 未解决 → reviser          ← 继续改
            → 上限 → finalize
        → 无高置信度 issue → finalize   ← 不值得改
"""

from langgraph.graph import END, StateGraph

from src.graph.state import AgentState
from src.graph.nodes import (
    retrieve_node,
    plan_node,
    multi_hop_retrieve_node,
    generator_node,
    critic_node,
    reviser_node,
    verify_node,
    finalize_node,
)
from src.graph.routers import route_after_plan, route_after_critic, route_after_verify


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    # ── 注册节点 ──────────────────────────────────────────────────────
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("plan", plan_node)
    builder.add_node("multi_hop_retrieve", multi_hop_retrieve_node)
    builder.add_node("generator", generator_node)
    builder.add_node("critic", critic_node)
    builder.add_node("reviser", reviser_node)
    builder.add_node("verify", verify_node)
    builder.add_node("finalize", finalize_node)

    # ── 入口 ──────────────────────────────────────────────────────────
    builder.set_entry_point("retrieve")

    # ── 检索 → 规划 → 分支 → 生成 ──────────────────────────────────
    builder.add_edge("retrieve", "plan")
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {"generator": "generator", "multi_hop_retrieve": "multi_hop_retrieve"},
    )
    builder.add_edge("multi_hop_retrieve", "generator")

    # ── 生成 → 审查(full) → 分支 ────────────────────────────────────
    builder.add_edge("generator", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"finalize": "finalize", "reviser": "reviser"},
    )

    # ── 修订 → 定向验证 → 分支 ──────────────────────────────────────
    builder.add_edge("reviser", "verify")
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {"critic": "critic", "reviser": "reviser", "finalize": "finalize"},
    )

    # ── 输出 ──────────────────────────────────────────────────────────
    builder.add_edge("finalize", END)

    return builder.compile()
