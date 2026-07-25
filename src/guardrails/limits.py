"""护栏模块 — 运行时限制强制执行

集中管理辩论轮次、修正轮次和检索跳数的上限检查，
为条件路由和监控日志提供单一真相来源。
"""

from config.settings import MAX_CORRECTION_ROUNDS, MAX_DEBATE_ROUNDS, MAX_RETRIEVAL_HOPS


def can_continue_debate(debate_round: int) -> bool:
    """检查是否可以继续下一轮辩论。"""
    return debate_round < MAX_DEBATE_ROUNDS


def can_continue_correction(correction_round: int) -> bool:
    """检查是否可以继续修正答案。"""
    return correction_round < MAX_CORRECTION_ROUNDS


def can_retrieve_more(retrieval_hops: int) -> bool:
    """检查是否可以继续多跳检索。"""
    return retrieval_hops < MAX_RETRIEVAL_HOPS


def debate_rounds_exhausted(debate_round: int) -> bool:
    """检查辩论轮次是否已用尽。"""
    return debate_round >= MAX_DEBATE_ROUNDS


def correction_rounds_exhausted(correction_round: int) -> bool:
    """检查修正轮次是否已用尽。"""
    return correction_round >= MAX_CORRECTION_ROUNDS
