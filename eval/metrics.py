"""自定义评估指标 — 用 DeepSeek API 替代 RAGAS，避免依赖冲突

实现 RAGAS 同等效果的 4 个指标：
- faithfulness (忠实度): 答案是否基于检索上下文
- answer_relevancy (答案相关性): 答案与问题的匹配度
- context_precision (上下文精确率): 检索结果的信号噪声比
- context_recall (上下文召回率): 检索是否覆盖了 ground truth 要点

设计原则：LLM 只做「逐条分类判定」，score 由程序按比率计算。
自由打分（"请给 0-1 分"）在同输入下波动可达 ±0.2，而分类判定
（supported/unsupported、covered/not covered）稳定性高一个量级，
多次测量结果可复现。所有指标返回的 score 均为 程序算出的比率，
不是 LLM 自报分。
"""

import json
import sys

from openai import OpenAI
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CRITIC_MODEL


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client


# ============================================================================
# 上下文长度控制 — 防止评估 judge 输入超限
# ============================================================================

# 单篇文档送入评估 judge 的最大字符数（parent chunk 原文可达 3000）
MAX_CTX_CHARS_PER_DOC = 2000
# 所有文档总字符预算：超出后等比压缩。
# 9-12 篇 parent chunk 全文可达 3 万+ 字符，实测会触发网关返回空 choices。
MAX_CTX_CHARS_TOTAL = 22000


def _truncate_contexts(contexts: list[str]) -> list[str]:
    """按篇截断 + 总预算等比压缩，防止评估 judge 上下文超限。"""
    per_doc = [ctx[:MAX_CTX_CHARS_PER_DOC] for ctx in contexts]
    total = sum(len(c) for c in per_doc)
    if total <= MAX_CTX_CHARS_TOTAL:
        return per_doc
    ratio = MAX_CTX_CHARS_TOTAL / total
    return [c[: max(200, int(len(c) * ratio))] for c in per_doc]


def _call_judge_json(system_prompt: str, user_prompt: str, max_retries: int = 2) -> dict:
    """调用 judge 模型并解析 JSON 输出。

    对三类失败统一兜底并重试：
    - API 层异常（限流/超时/5xx）
    - 响应结构异常（choices 为空，常见于上下文超限或网关错误）
    - JSON 解析失败
    重试仍失败时返回 {"_parse_error": True, ...}，调用方按 0 分处理。
    """
    import time

    client = _get_client()
    last_err = "unknown"

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=CRITIC_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            choices = getattr(response, "choices", None)
            if not choices:
                # choices 为 None 或 []：打印响应原文便于诊断
                last_err = f"empty choices, raw response: {str(response)[:400]}"
            else:
                content = choices[0].message.content
                try:
                    return json.loads(content)
                except (json.JSONDecodeError, TypeError) as e:
                    last_err = f"JSON decode: {e}, content head: {str(content)[:200]}"
        except Exception as e:  # openai.APIError 及一切网络层异常
            last_err = f"{type(e).__name__}: {e}"

        if attempt < max_retries:
            time.sleep(2 * (attempt + 1))

    print(f"[metrics] judge 调用失败（重试 {max_retries} 次后仍失败）: {last_err}", file=sys.stderr)
    return {"_parse_error": True, "reasoning": f"judge call failed: {last_err}"}


# ============================================================================
# 1. Faithfulness（忠实度）— 逐条断言判定，score = supported / total
# ============================================================================

FAITHFULNESS_PROMPT = """你是一个严格的评估专家。请评估以下"生成的答案"是否完全基于"提供的上下文"。

评估步骤：
1. 从答案中逐句提取所有事实性断言（只提取可验证的事实陈述；承接句、观点性总结不计入）
2. 对每条断言，判断它在上下文中的支持情况：
   - supported：上下文中有直接原文或明确同义表述
   - unsupported：上下文中找不到该断言的依据
   - contradicted：上下文中有相反表述
3. 注意：答案中的 [来源 N] 编号与上下文的 [文档 N] 编号体系无关，请只按内容核对

请输出纯JSON格式（不要markdown包裹）：
{{
    "claims": [
        {{
            "claim": "<答案中的断言原文>",
            "status": "supported" | "unsupported" | "contradicted",
            "evidence": "<上下文中的依据原文；无依据时填空字符串>"
        }}
    ],
    "reasoning": "<简要总结，中文>"
}}"""


def evaluate_faithfulness(question: str, answer: str, contexts: list[str]) -> dict:
    """评估答案忠实度。score = supported 断言数 / 总断言数（程序计算）。"""
    contexts = _truncate_contexts(contexts)
    context_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {ctx}" for i, ctx in enumerate(contexts)
    )

    user_prompt = f"""问题: {question}

提供的上下文:
{context_text}

生成的答案:
{answer}

请逐条提取断言并判定支持情况。"""

    result = _call_judge_json(FAITHFULNESS_PROMPT, user_prompt)
    claims = result.get("claims") or []
    if not isinstance(claims, list):
        claims = []

    total = len(claims)
    supported = sum(
        1 for c in claims
        if isinstance(c, dict) and str(c.get("status", "")).lower() == "supported"
    )
    contradicted = sum(
        1 for c in claims
        if isinstance(c, dict) and str(c.get("status", "")).lower() == "contradicted"
    )
    score = supported / total if total > 0 else 0.0

    return {
        "score": round(score, 4),
        "hallucination_count": total - supported,
        "total_claims": total,
        "contradicted_count": contradicted,
        "claims": claims,
        "reasoning": result.get("reasoning", ""),
    }


# ============================================================================
# 2. Answer Relevancy（答案相关性）— 信息点覆盖判定，score = covered / total
# ============================================================================

RELEVANCY_PROMPT = """你是一个严格的评估专家。请评估"生成的答案"与"用户问题"的相关性。

评估步骤：
1. 分析用户问题，列出完整回答它所需要覆盖的信息点清单（2-6 个）
2. 逐一判断答案是否覆盖了每个信息点（covered=true/false）
3. 列出答案中与问题无关的冗余内容（off_topic，如有）

请输出纯JSON格式（不要markdown包裹）：
{{
    "required_points": [
        {{"point": "<回答该问题所需的信息点>", "covered": true | false}}
    ],
    "off_topic": ["<答案中的偏题内容>"],
    "reasoning": "<简要说明，中文>"
}}"""


def evaluate_answer_relevancy(question: str, answer: str) -> dict:
    """评估答案相关性。score = 被覆盖信息点数 / 信息点总数（程序计算）。

    off_topic 仅作报告，不参与计分——覆盖率与冗余度是两个维度，
    混合计分会降低可解释性。
    """
    user_prompt = f"""用户问题: {question}

生成的答案:
{answer}

请列出所需信息点并逐条判定覆盖情况。"""

    result = _call_judge_json(RELEVANCY_PROMPT, user_prompt)
    points = result.get("required_points") or []
    if not isinstance(points, list):
        points = []

    total = len(points)
    covered = sum(
        1 for p in points if isinstance(p, dict) and bool(p.get("covered"))
    )
    score = covered / total if total > 0 else 0.0
    off_topic = result.get("off_topic") or []

    return {
        "score": round(score, 4),
        "covered_points": covered,
        "total_points": total,
        "off_topic": off_topic if isinstance(off_topic, list) else [],
        "reasoning": result.get("reasoning", ""),
    }


# ============================================================================
# 3. Context Precision（上下文精确率）— 逐文档相关判定，score = relevant / total
# ============================================================================

PRECISION_PROMPT = """你是一个严格的评估专家。请评估"检索到的文档片段"中，有多少真正与"用户问题"相关。

评估步骤：
1. 分析用户问题需要什么信息
2. 逐个检查每个检索到的文档片段，判断是否包含对回答问题有用的信息
   - 部分相关（含少量可用信息）也算 relevant=true
   - 仅主题沾边但无可用信息的判 relevant=false
3. documents 数组必须覆盖所有片段，不得遗漏

请输出纯JSON格式（不要markdown包裹）：
{{
    "documents": [
        {{"doc_id": <片段编号, 整数>, "relevant": true | false, "reason": "<一句话>"}}
    ],
    "reasoning": "<简要说明，中文>"
}}"""


def evaluate_context_precision(question: str, contexts: list[str]) -> dict:
    """评估上下文精确率。score = 相关片段数 / 片段总数（程序计算）。

    分母固定为 len(contexts)：judge 漏判的片段按不相关处理（保守口径）。
    """
    docs_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {ctx[:800]}" for i, ctx in enumerate(contexts)
    )

    user_prompt = f"""用户问题: {question}

检索到的文档片段（共{len(contexts)}个）:
{docs_text}

请逐片段判定相关性。"""

    result = _call_judge_json(PRECISION_PROMPT, user_prompt)
    documents = result.get("documents") or []
    if not isinstance(documents, list):
        documents = []

    relevant = sum(
        1 for d in documents if isinstance(d, dict) and bool(d.get("relevant"))
    )
    total = len(contexts)
    score = relevant / total if total > 0 else 0.0

    return {
        "score": round(score, 4),
        "relevant_count": relevant,
        "total_documents": total,
        "documents": documents,
        "reasoning": result.get("reasoning", ""),
    }


# ============================================================================
# 4. Context Recall（上下文召回率）— GT 要点覆盖判定，score = covered / total
# ============================================================================

RECALL_PROMPT = """你是一个严格的评估专家。请评估"检索到的文档片段"是否覆盖了"参考答案"中的关键信息点。

评估步骤：
1. 从参考答案中提取关键信息点（3-8 个事实性要点）
2. 逐一判断每个信息点是否能在检索文档中找到依据（明确同义表述也算覆盖）
3. 记录未能覆盖的信息点

请输出纯JSON格式（不要markdown包裹）：
{{
    "points": [
        {{
            "point": "<参考答案中的信息点>",
            "covered": true | false,
            "evidence": "<检索文档中的对应内容；未覆盖时填空字符串>"
        }}
    ],
    "reasoning": "<简要说明，中文>"
}}"""


def evaluate_context_recall(question: str, contexts: list[str], ground_truth: str) -> dict:
    """评估上下文召回率。score = 被覆盖要点数 / 要点总数（程序计算）。"""
    docs_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {ctx[:800]}" for i, ctx in enumerate(contexts)
    )

    user_prompt = f"""用户问题: {question}

参考答案（Ground Truth）:
{ground_truth}

检索到的文档片段（共{len(contexts)}个）:
{docs_text}

请提取参考答案要点并逐条判定覆盖情况。"""

    result = _call_judge_json(RECALL_PROMPT, user_prompt)
    points = result.get("points") or []
    if not isinstance(points, list):
        points = []

    total = len(points)
    covered = sum(
        1 for p in points if isinstance(p, dict) and bool(p.get("covered"))
    )
    score = covered / total if total > 0 else 0.0
    missing = [
        p.get("point", "") for p in points
        if isinstance(p, dict) and not bool(p.get("covered"))
    ]

    return {
        "score": round(score, 4),
        "covered_points": covered,
        "total_points": total,
        "missing_info": missing,
        "reasoning": result.get("reasoning", ""),
    }


# ============================================================================
# 综合评估入口
# ============================================================================

def evaluate_all(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict:
    """对单个问题运行全部 4 个评估指标。

    注意：context_precision / context_recall 只依赖 question + contexts，
    与 answer 无关。对同一份 contexts 重复调用只会得到噪声，
    调用方应对每份检索结果只评一次（见 run_eval_debate.py）。

    Returns:
        {
            "faithfulness": {"score": ..., ...},
            "answer_relevancy": {"score": ..., ...},
            "context_precision": {"score": ..., ...},
            "context_recall": {"score": ..., ...},
        }
    """
    results = {}

    results["faithfulness"] = evaluate_faithfulness(question, answer, contexts)
    results["answer_relevancy"] = evaluate_answer_relevancy(question, answer)
    results["context_precision"] = evaluate_context_precision(question, contexts)
    results["context_recall"] = evaluate_context_recall(question, contexts, ground_truth)

    return results
