"""Critic Agent — 质量审查与裁决

融合了原来 Skeptic（逐条审查）+ Judge（裁决）+ Validator（核验）三个角色。
一人全流程：审查 → 判定 → 给出修改指导。

关键设计原则（基于过往三轮 eval 数据的教训）：
- 已经准确完整的答案，不要为追求完美而制造问题（防过修正退化）
- "遗漏"只标记问题核心答案维度的缺失，不纠缠边缘内容
- reject 必须引用"与文献矛盾"或"核心结论无据"级证据
- 每条 issue 标注 severity（critical/minor），Reviser 据此优先排序
"""

import json

from config.settings import JUDGE_MODEL
from src.utils.llm import call_llm_with_json

# ============================================================================
# System Prompt
# ============================================================================

CRITIC_SYSTEM_PROMPT = """
你是一位极其严格且客观的学术期刊审稿人。你的唯一任务是对照参考文献，对作者提交的答案进行逐条事实核查与完整性审查。

## 核心原则

1. **只管事实，不管文笔**：你不负责评价语言是否优美、结构是否巧妙。只要事实准确且有文献支撑，即使写得像大白话也不算问题。
2. **严禁主观推测**：必须严格区分"文献有写"与"我推断如此"。文献只提到 A 和 B，绝不能判定答案推断的"A 导致 B"为有据，除非文献明确指出了因果关系。
3. **零外部知识**：判定依据仅限于提供的参考文献，绝对不使用你自身的知识库去证实或证伪答案。

## 审查流程

1. **提取断言**：逐句提取答案中的事实性陈述（如具体机制、数值、效果、对比结论等）。忽略承接句和过渡词。
2. **核查状态**：对照参考文献，判断每个断言的状态：
   - **supported**：文献中有直接原文或明确同义表述。
   - **unsupported**：文献中未提及（无据断言或过度推断）。
   - **contradicted**：文献中有相反的表述。
3. **核查完整性**：文献中是否存在与问题直接相关、对理解核心结论有实质性影响的关键信息被遗漏？（边缘细节不算遗漏）。

## 审稿界限（以下情况绝对不算问题，禁止输出）

- **详略不同**：答案概括性描述，文献有详细论述（反之亦然），只要核心意思一致即为 supported。
- **条件分支**：通用规则与特定条件下的例外并存，不构成矛盾。
- **非核心遗漏**：不影响回答问题主旨的次要实验参数或背景细节。
- **不同来源差异**：答案引用了文献 A 的结论，而文献 B 有不同结论，只要标注无误，不算矛盾。

## 置信度评分规则

对每一个输出的 issue，按以下客观文本特征标注 confidence（0.0-1.0 的浮点数）：
- **0.90-1.0（字面确证）**：文献原文与答案断言存在直接互斥的数值、相反的动作/状态表述，或文献中完全找不到对应痕迹。仅凭字面对比即可 100% 确认问题存在。
- **0.75-0.89（逻辑确证）**：单看字面不直接互斥，但放入同一逻辑框架后无法同时成立。例如文献描述的流程明确包含了某步骤，答案却说"不需要该步骤"。
- **0.60-0.74（推断/歧义型）**：答案在当前语境下可能存在问题，但矛盾的根源更有可能是答案遗漏了限制条件或表述有歧义，而非故意犯错。

## 输出格式

必须严格输出合法 JSON，不要包含任何 Markdown 标记或解释性废话。

若答案准确无误、没有需要修改的问题，直接输出：
{"verdict": "accept", "issues": []}

若存在问题，严格按以下 schema 输出：

{
  "verdict": "revise" | "reject",
  "issues": [
    {
      "claim": "<答案中存在问题的具体文本原文>",
      "status": "unsupported" | "contradicted" | "missing",
      "confidence": 0.0-1.0,
      "evidence": "<文献中的相关原文摘录，用于证明问题存在。如果是missing，填文献中遗漏的关键信息原文>",
      "fix": "<具体的修改指示：应删除什么、改为引用什么、从文献哪里补充。作者将按此执行>"
    }
  ]
}
"""


# ============================================================================
# Call Function
# ============================================================================

def call_critic(
    query: str,
    answer: str,
    docs: list[dict],
    review_round: int,
) -> dict:
    """审查答案并给出裁决。

    Args:
        query: 用户问题。
        answer: 待审查的答案（draft 或修订后版本）。
        docs: 检索到的文献（Critic 独立核查依据）。
        review_round: 当前审查轮次（1-based）。

    Returns:
        {"verdict": str, "confidence": float, "issues": [dict],
         "improvements": str, "reasoning": str}
    """
    from src.utils.docs import format_docs

    context = format_docs(docs)

    round_hint = ""
    if review_round >= 3:
        round_hint = "\n这是第多次审查。如果答案没有与文献矛盾的严重事实错误，倾向于 accept 以避免过度修正。"
    elif review_round >= 2:
        round_hint = "\n这是第二轮审查。如果之前的修改已解决主要问题且无新增错误，倾向于 accept。小瑕疵可以放过。"

    user_prompt = f"""## 用户问题
{query}

## 待审查的答案
{answer}

## 参考文献
{context}

## 审查轮次
第 {review_round} 轮{round_hint}

请逐条审查答案的事实准确性，判断是否达到发表标准。输出 JSON。"""

    try:
        result = call_llm_with_json(
            model_name=JUDGE_MODEL,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        return {
            "verdict": result.get("verdict", "revise"),
            "confidence": float(result.get("confidence", 0.5)),
            "issues": result.get("issues", []) or [],
            "improvements": result.get("improvements", ""),
            "reasoning": result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return {
            "verdict": "revise",
            "confidence": 0.5,
            "issues": [],
            "improvements": f"Critic JSON parse error: {e}",
            "reasoning": f"Technical error: {e}",
        }


# ============================================================================
# Verify Mode — 定向检查 issues 是否已被 Reviser 解决
# ============================================================================

VERIFY_SYSTEM_PROMPT = """你是学术期刊的副主编。你的任务是**定向检查**：上一轮审稿人指出的具体问题，在作者修改后的答案中是否已经被妥善解决。

## 与 Full Review 的区别

Full Review 要对答案做全面的、从零开始的审查。你现在只需要**对照给定的 issues 列表，逐条判断是否已被修改**。不要发现新问题——即便你看到了其他可改进之处，也不要在这里提出。

## 判定标准

对每一个 issue，输出：

- **resolved**：答案中该问题的对应文本已被正确修正。修正后的表述与文献一致，或已补充所需信息。
- **unresolved**：答案中该问题仍然存在，或者修改后的表述引入了新的错误。
- **partially_resolved**：问题部分解决但不够彻底（如遗漏了证据中的关键定量数据、修正方向正确但引用缺失）。

## 输出格式

严格输出 JSON：

{
  "all_resolved": true | false,
  "details": [
    {
      "issue_index": <原 issues 列表中的索引, 从 0 开始>,
      "resolved": true | false | "partially",
      "reason": "<1-2 句中文判定理由>"
    }
  ]
}"""


def call_critic_verify(
    query: str,
    answer: str,
    previous_issues: list[dict],
    docs: list[dict],
) -> dict:
    from src.utils.docs import format_docs

    context = format_docs(docs)

    issues_text = ""
    for i, iss in enumerate(previous_issues):
        issues_text += (
            f"[{i}] claim: {iss.get('claim', '')}\n"
            f"    status: {iss.get('status', '?')}, severity: {iss.get('severity', '?')}, confidence: {iss.get('confidence', 0):.2f}\n"
            f"    fix: {iss.get('fix', '')}\n\n"
        )

    user_prompt = f"""## 用户问题
{query}

## 修改后的答案
{answer}

## 待检查的问题列表（来自上一轮审稿）
{issues_text}

## 参考文献
{context}

请逐条检查以上问题是否已被解决。输出 JSON。"""

    try:
        result = call_llm_with_json(
            model_name=JUDGE_MODEL,
            system_prompt=VERIFY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        return {
            "all_resolved": bool(result.get("all_resolved", False)),
            "details": result.get("details", []) or [],
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return {
            "all_resolved": False,
            "details": [
                {"issue_index": -1, "resolved": False, "reason": f"Verify JSON error: {e}"}
            ],
        }
