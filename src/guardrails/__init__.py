"""护栏模块 — 运行时限制与安全检查"""

from src.guardrails.limits import (
    can_continue_debate,
    can_continue_correction,
    can_retrieve_more,
    debate_rounds_exhausted,
    correction_rounds_exhausted,
)

__all__ = [
    "can_continue_debate",
    "can_continue_correction",
    "can_retrieve_more",
    "debate_rounds_exhausted",
    "correction_rounds_exhausted",
]
