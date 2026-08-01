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
import re
import sys

from config.settings import (
    JUDGE_MODEL,
    CRITIC_TOOL_VERIFY_ENABLED,
    CRITIC_TOOL_VERIFY_CONF_THRESHOLD,
    CRITIC_TOOL_VERIFY_MAX_CLAIMS,
)
from src.utils.llm import call_llm_with_json, call_llm_tool_loop

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

必须严格输出合法 JSON，不要包含任何 Markdown 标记或解释性废话。所有字段均为必填，不得省略。

若答案准确无误、没有需要修改的问题，直接输出：
{"verdict": "accept", "confidence": 0.0-1.0, "issues": []}

若存在问题，严格按以下 schema 输出：

{
  "verdict": "revise" | "reject",
  "confidence": 0.0-1.0,
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

注意区分两级 confidence：顶层 confidence 是你对**整体裁决**的把握（accept 时也必须输出）；issue 级 confidence 是该问题确实成立的把握，只有 ≥0.8 的 issue 会被交给作者修改——请如实评估，不要为了让答案被修改而虚报高置信度。
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
            return_raw=True,
        )
        raw_output = result["raw"]
        result = result["parsed"]
        if "confidence" not in result:
            print(
                "[critic] 警告: 裁决 JSON 缺少 confidence 字段，回退默认值 0.5",
                file=sys.stderr,
            )
        ruling = {
            "verdict": result.get("verdict", "revise"),
            "confidence": float(result.get("confidence", 0.5)),
            "issues": result.get("issues", []) or [],
            "improvements": result.get("improvements", ""),
            "reasoning": result.get("reasoning", ""),
            # 模型原始输出原文（完整保留，供控制台全量展示）
            "raw_output": raw_output,
        }
        if CRITIC_TOOL_VERIFY_ENABLED and ruling["verdict"] != "accept":
            ruling = _tool_verify_issues(query, ruling)
        return ruling
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return {
            "verdict": "revise",
            "confidence": 0.5,
            "issues": [],
            "improvements": f"Critic JSON parse error: {e}",
            "reasoning": f"Technical error: {e}",
        }


# ============================================================================
# Tool-Assisted Verification — 存疑断言的工具核查（拦截假阳性）
# ============================================================================
# Critic 只能对照「已检索到的文档集」判 unsupported——断言其实真实、只是
# 证据没被检索到时，就会被误判并触发有害修订（q001 型退化）。这里对低置信度
# 的 unsupported/contradicted 断言，先用工具在全语料查证，再决定是否保留。

CLAIM_VERIFIER_SYSTEM_PROMPT = """你是事实核查员。给你一条从学术答案中提取的断言，你的任务是判断它在本地语料库（论文/框架文档/博客）中**是否有依据**。

## 工作流程

1. 调用工具检索：paper_search 查论文、concept_query 查概念定义、framework_doc_search 查框架文档。可多次调用、换关键词。
2. **工具调用最多 2 轮**——拿到初步材料后就比对，不要无限深挖。检索关键词用断言中的核心术语（英文术语优先），不要整句复制。
3. 仔细比对检索到的原文与断言：
   - 有直接原文或明确同义表述 → supported
   - 找不到依据，或与原文相反 → unsupported

## 输出

完成检索后，只输出一行 JSON（不要 markdown 包裹）：
{"verdict": "supported" | "unsupported", "reason": "一句中文理由"}"""


def _verify_claim_with_tools(query: str, claim: str) -> dict:
    """用工具在全语料中核查单条断言。

    Returns:
        {"verdict": "supported"|"unsupported"|"unknown",
         "reason": str, "fetched_chunk_ids": [str]}
    """
    from src.mcp_server.tools import (
        PAPER_SEARCH_SCHEMA,
        CONCEPT_QUERY_SCHEMA,
        FRAMEWORK_DOC_SEARCH_SCHEMA,
    )

    user_prompt = f"""## 原始问题
{query}

## 待核查断言
{claim}

请调用工具检索语料，判断该断言是否有依据。"""

    try:
        result = call_llm_tool_loop(
            model_name=JUDGE_MODEL,
            system_prompt=CLAIM_VERIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=[PAPER_SEARCH_SCHEMA, CONCEPT_QUERY_SCHEMA, FRAMEWORK_DOC_SEARCH_SCHEMA],
            max_rounds=3,
            temperature=0.0,
        )
    except Exception as e:
        print(f"[critic] 工具核查调用失败: {e}", file=sys.stderr)
        return {"verdict": "unknown", "reason": f"tool verify error: {e}",
                "fetched_chunk_ids": []}

    text = result.get("text", "")

    # 解析 verdict：优先 JSON，退化到关键词匹配（先 unsupported 后 supported，
    # 防止 "unsupported" 中的子串被误判为 supported）
    verdict = "unknown"
    parsed = None
    json_candidate = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(json_candidate)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r'"(unsupported|supported)"', text) or re.search(
            r"\b(unsupported|supported)\b", text
        )
        if m:
            parsed = {"verdict": m.group(1)}
    if isinstance(parsed, dict):
        v = str(parsed.get("verdict", "")).lower()
        if v in ("supported", "unsupported"):
            verdict = v

    # reason 优先取解析出的 JSON 内层字段，避免把整段 JSON 原文塞进去
    reason = text[:200]
    if isinstance(parsed, dict) and parsed.get("reason"):
        reason = str(parsed["reason"])[:200]

    # 从工具调用日志中提取命中的 chunk_id（供并入检索集，守评测口径）
    fetched: list[str] = []
    tool_trace: list[dict] = []
    for entry in result.get("tool_log", []):
        try:
            payload = json.loads(entry.get("result", "{}"))
        except (json.JSONDecodeError, TypeError):
            payload = {}
        hits = 0
        for key in ("results", "results_a", "results_b"):
            results = payload.get(key, []) or []
            hits += len(results)
            for r in results:
                cid = r.get("chunk_id")
                if cid and cid not in fetched:
                    fetched.append(cid)
        # 压缩记录每次工具调用（名称/参数/命中数），供控制台展示完整核查过程
        tool_trace.append({
            "name": entry.get("name", "?"),
            "args": entry.get("args", {}),
            "hits": hits,
        })

    return {
        "verdict": verdict,
        "reason": reason,
        "fetched_chunk_ids": fetched[:8],
        "tool_trace": tool_trace,
    }


def _tool_verify_issues(query: str, ruling: dict) -> dict:
    """对低置信度的 unsupported/contradicted 断言做工具核查。

    核查改判 supported 的 issue 视为假阳性，直接从 issues 中移除；
    若所有 issue 均被拦截，裁决回退为 accept（修订依据已不存在）。
    """
    issues = ruling.get("issues", [])
    candidates = [
        i for i in issues
        if i.get("status") in ("unsupported", "contradicted")
        and float(i.get("confidence", 0) or 0) < CRITIC_TOOL_VERIFY_CONF_THRESHOLD
    ][:CRITIC_TOOL_VERIFY_MAX_CLAIMS]

    if not candidates:
        ruling["tool_verifications"] = []
        return ruling

    verifications = []
    intercepted_ids = set()
    for iss in candidates:
        v = _verify_claim_with_tools(query, iss.get("claim", ""))
        verifications.append({
            "claim": iss.get("claim", "")[:200],
            "verdict": v["verdict"],
            "reason": v["reason"],
            "fetched_chunk_ids": v["fetched_chunk_ids"],
            "tool_trace": v.get("tool_trace", []),
        })
        if v["verdict"] == "supported":
            intercepted_ids.add(id(iss))

    if intercepted_ids:
        ruling["issues"] = [i for i in issues if id(i) not in intercepted_ids]
        if not ruling["issues"]:
            ruling["verdict"] = "accept"
            ruling["reasoning"] = (
                f"{ruling.get('reasoning', '')} "
                f"（工具核查：{len(intercepted_ids)} 条存疑断言在全语料中找到依据，"
                f"判定为假阳性，改判 accept）"
            ).strip()

    ruling["tool_verifications"] = verifications
    return ruling


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
