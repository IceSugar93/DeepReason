"""图条件边路由 — 3-Agent 审查-修订引擎（含 Verify 定向检查）

三个路由函数：
- route_after_plan: 决定是否需要多跳检索
- route_after_critic: 裁决后分支（accept → 结束 / revise/reject → 过滤后进修订）
- route_after_verify: 定向验证后分支（全部解决 → 重审 / 还有问题 → 继续修订）
"""

from config.settings import MAX_CORRECTION_ROUNDS as MAX_REVISION_ROUNDS

MAX_REVIEW_ROUNDS = MAX_REVISION_ROUNDS + 1


def route_after_plan(state: dict) -> str:
    if state.get("complexity") == "multi_hop" and state.get("sub_questions"):
        return "multi_hop_retrieve"
    return "generator"


def route_after_critic(state: dict) -> str:
    """Critic 审查后的分支路由。

    - accept → 输出最终结果
    - revise/reject：
        - 无高置信度 issue → 直接输出（没有值得改的东西）
        - 有高置信度 issue + 未超上限 → 进入修订
        - 达到上限 → 强制输出
    """
    verdict = state.get("critic_ruling", {}).get("verdict", "revise")
    previous_issues = state.get("previous_issues", [])
    revision_round = state.get("revision_round", 0)
    review_round = state.get("review_round", 0)

    if verdict == "accept":
        return "finalize"

    # 没有高置信度 issue：不值得修订，直接输出
    if not previous_issues:
        return "finalize"

    if revision_round < MAX_REVISION_ROUNDS and review_round < MAX_REVIEW_ROUNDS:
        return "reviser"

    return "finalize"


def route_after_verify(state: dict) -> str:
    """定向验证后的分支路由。

    - 全部解决 + 审查轮次未满 → 回到 Critic(full) 重新全面审查
    - 有未解决 + 修订轮次未满 → 回到 Reviser 继续改
    - 达到上限 → 强制输出
    """
    verify_result = state.get("verify_result", {})
    all_resolved = verify_result.get("all_resolved", False)
    revision_round = state.get("revision_round", 0)
    review_round = state.get("review_round", 0)

    if all_resolved and review_round < MAX_REVIEW_ROUNDS:
        return "critic"

    if not all_resolved and revision_round < MAX_REVISION_ROUNDS:
        return "reviser"

    return "finalize"
