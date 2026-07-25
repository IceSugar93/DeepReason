"""Planner Agent — 问题拆解与复杂度判断

分析用户问题，判断是否需要拆解为子问题，并规划检索策略。
"""

import json

from config.settings import PLANNER_MODEL
from src.utils.llm import call_llm_with_json

# ============================================================================
# System Prompt
# ============================================================================

PLANNER_SYSTEM_PROMPT = """你是一个研究推理引擎的查询规划专家。你的任务是分析用户问题，判断复杂度并决定检索策略。

## 判断规则

1. **简单问题（simple）**：事实性问题，从 1-2 个文档片段即可完整回答。
   - 例："BGE-M3 的向量维度是多少？"、"RAG 的全称是什么？"

2. **多跳问题（multi_hop）**：需要综合多个来源、多步推理、对比分析或多维度回答。拆解为 2-4 个独立子问题。
   - 例："AsyncLM 的异步机制如何工作？有哪些挑战？" → 子问题不应相互依赖
   - 子问题按逻辑顺序排列：先背景、再机制、后分析

## 拆解原则

- 每个子问题必须能独立回答（不依赖其他子问题的答案）
- 子问题覆盖原问题的所有维度，不遗漏
- 只拆解真正需要多步推理的问题，不要过度拆解

## 输出格式

严格输出 JSON：

{
  "complexity": "simple" | "multi_hop",
  "reasoning": "1 句话说明判断依据",
  "sub_questions": ["子问题1", "子问题2", ...]
}

对于 simple 问题，sub_questions 为空数组 []。"""


# ============================================================================
# Call Function
# ============================================================================

def call_planner(query: str, doc_summaries: list[str]) -> dict:
    """调用 Planner 分析问题复杂度并拆解子问题。

    Args:
        query: 用户原始问题。
        doc_summaries: 检索到的文档标题/摘要列表。

    Returns:
        {"complexity": str, "reasoning": str, "sub_questions": [str]}
    """
    if doc_summaries:
        context_lines = "\n".join(
            f"- [{i+1}] {s}" for i, s in enumerate(doc_summaries[:8])
        )
    else:
        context_lines = "（暂无检索结果，仅基于问题本身判断）"

    user_prompt = f"""## 用户问题
{query}

## 检索到的相关文档摘要
{context_lines}

请分析问题复杂度，判断是否需要拆解子问题。输出 JSON。"""

    try:
        result = call_llm_with_json(
            model_name=PLANNER_MODEL,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        return {
            "complexity": result.get("complexity", "simple"),
            "reasoning": result.get("reasoning", ""),
            "sub_questions": result.get("sub_questions", []),
        }
    except (json.JSONDecodeError, KeyError) as e:
        return {
            "complexity": "simple",
            "reasoning": f"Planner JSON parse error: {e}",
            "sub_questions": [],
        }
