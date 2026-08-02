"""护栏模块 — 运行时限制、收敛检测与安全检查"""

from src.guardrails.limits import (
    AGENT_TIMEOUT,
    can_continue_debate,
    can_continue_correction,
    can_retrieve_more,
    debate_rounds_exhausted,
    correction_rounds_exhausted,
)
from src.guardrails.convergence import check_answer_convergence, answer_similarity
from src.guardrails.safety import detect_risk, build_answer_annotations

__all__ = [
    "AGENT_TIMEOUT",
    "can_continue_debate",
    "can_continue_correction",
    "can_retrieve_more",
    "debate_rounds_exhausted",
    "correction_rounds_exhausted",
    "check_answer_convergence",
    "answer_similarity",
    "detect_risk",
    "build_answer_annotations",
]
