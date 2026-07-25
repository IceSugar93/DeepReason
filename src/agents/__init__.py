"""多 Agent 推理引擎 — 3 个核心角色模块（2026-07-24 架构简化）

- Planner:   问题拆解与检索策略
- Generator: 基于文献生成答案草稿
- Critic:    审查答案质量，给出 accept/revise/reject 判定与修改指导
- Reviser:   根据 Critic 意见修订答案
"""

from src.agents.planner import call_planner, PLANNER_SYSTEM_PROMPT
from src.agents.generator import call_generator, GENERATOR_SYSTEM_PROMPT
from src.agents.critic import call_critic, call_critic_verify
from src.agents.reviser import call_reviser, REVISER_SYSTEM_PROMPT
from src.utils.docs import format_docs

__all__ = [
    # Planner
    "call_planner",
    "PLANNER_SYSTEM_PROMPT",
    # Generator
    "call_generator",
    "GENERATOR_SYSTEM_PROMPT",
    # Critic
    "call_critic",
    "call_critic_verify",
    # Reviser
    "call_reviser",
    "REVISER_SYSTEM_PROMPT",
    # Helpers
    "format_docs",
]
