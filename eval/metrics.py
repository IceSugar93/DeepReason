"""自定义评估指标 — 用 DeepSeek API 替代 RAGAS，避免依赖冲突

实现 RAGAS 同等效果的 4 个指标：
- faithfulness (忠实度): 答案是否基于检索上下文
- answer_relevancy (答案相关性): 答案与问题的匹配度
- context_precision (上下文精确率): 检索结果的信号噪声比
- context_recall (上下文召回率): 检索是否覆盖了 ground truth 要点
"""

import json

from openai import OpenAI
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, JUDGE_MODEL


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client


# ============================================================================
# 1. Faithfulness（忠实度）— 答案是否基于上下文，有无幻觉
# ============================================================================

FAITHFULNESS_PROMPT = """你是一个严格的评估专家。请评估以下"生成的答案"是否完全基于"提供的上下文"。

评估步骤：
1. 逐句检查生成答案中的每个断言
2. 判断每个断言是否能在上下文中找到依据
3. 如果某个断言在上下文中找不到依据，标记为"幻觉"
4. 综合所有断言，给出忠实度评分

评分标准：
- 1.0: 答案中的所有断言都可以在上下文中找到直接依据
- 0.7-0.9: 大部分断言有依据，少量细节推断（但合理）
- 0.4-0.6: 部分断言有依据，存在明显的无依据陈述
- 0.1-0.3: 大部分断言在上下文中找不到依据
- 0.0: 答案与上下文完全无关或完全编造

请输出纯JSON格式（不要markdown包裹）：
{{
    "score": <0.0到1.0的浮点数>,
    "hallucination_count": <幻觉断言数量>,
    "total_claims": <总断言数量>,
    "reasoning": "<简要说明扣分原因，中文>"
}}"""


def evaluate_faithfulness(question: str, answer: str, contexts: list[str]) -> dict:
    """评估答案忠实度。"""
    context_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {ctx}" for i, ctx in enumerate(contexts)
    )

    user_prompt = f"""问题: {question}

提供的上下文:
{context_text}

生成的答案:
{answer}

请评估该答案的忠实度。"""

    client = _get_client()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": FAITHFULNESS_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ============================================================================
# 2. Answer Relevancy（答案相关性）— 答案是否切题
# ============================================================================

RELEVANCY_PROMPT = """你是一个严格的评估专家。请评估"生成的答案"与"用户问题"的相关性。

评估步骤：
1. 分析用户问题的核心意图和所需信息点
2. 检查生成的答案是否覆盖了这些信息点
3. 检查答案中是否有冗余或不相关的内容
4. 综合给出相关性评分

评分标准：
- 1.0: 答案精准回应了问题的所有要点，无冗余内容
- 0.7-0.9: 答案回应了主要问题，少量次要内容偏题
- 0.4-0.6: 答案部分回应了问题，有较多偏题内容
- 0.1-0.3: 答案与问题大部分不相关
- 0.0: 答案完全避而不答或答非所问

请输出纯JSON格式（不要markdown包裹）：
{{
    "score": <0.0到1.0的浮点数>,
    "covered_points": <答案覆盖了问题几个信息点>,
    "total_points": <问题共有几个信息点>,
    "reasoning": "<简要说明，中文>"
}}"""


def evaluate_answer_relevancy(question: str, answer: str) -> dict:
    """评估答案相关性。"""
    user_prompt = f"""用户问题: {question}

生成的答案:
{answer}

请评估该答案与问题的相关性。"""

    client = _get_client()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": RELEVANCY_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ============================================================================
# 3. Context Precision（上下文精确率）— 检索到的文档有多少真正相关
# ============================================================================

PRECISION_PROMPT = """你是一个严格的评估专家。请评估"检索到的文档片段"中，有多少真正与"用户问题"相关。

评估步骤：
1. 分析用户问题需要什么信息
2. 逐个检查每个检索到的文档片段
3. 判断每个片段是否包含对回答问题有用的信息
4. 标记为"相关"或"不相关"
5. 计算相关片段占比

评分标准：
- 1.0: 所有检索到的片段都与问题相关
- 0.7-0.9: 大部分片段相关，少量噪声
- 0.4-0.6: 约一半片段相关
- 0.1-0.3: 大部分片段不相关
- 0.0: 所有片段都不相关

请输出纯JSON格式（不要markdown包裹）：
{{
    "score": <0.0到1.0的浮点数>,
    "relevant_count": <相关片段数>,
    "total_documents": <总片段数>,
    "reasoning": "<简要说明，中文>"
}}"""


def evaluate_context_precision(question: str, contexts: list[str]) -> dict:
    """评估上下文精确率。"""
    docs_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {ctx[:800]}" for i, ctx in enumerate(contexts)
    )

    user_prompt = f"""用户问题: {question}

检索到的文档片段（共{len(contexts)}个）:
{docs_text}

请评估这些检索结果的精确率。"""

    client = _get_client()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": PRECISION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ============================================================================
# 4. Context Recall（上下文召回率）— 检索是否覆盖了 ground truth 要点
# ============================================================================

RECALL_PROMPT = """你是一个严格的评估专家。请评估"检索到的文档片段"是否覆盖了"参考答案"中的关键信息点。

评估步骤：
1. 从参考答案中提取关键信息点（事实性断言）
2. 逐一检查每个信息点是否能在检索到的文档中找到
3. 统计被覆盖的信息点占比

评分标准：
- 1.0: 参考答案的所有关键信息点都能在检索文档中找到
- 0.7-0.9: 大部分信息点被覆盖
- 0.4-0.6: 约一半信息点被覆盖
- 0.1-0.3: 少量信息点被覆盖
- 0.0: 检索文档完全不包含参考答案的信息

请输出纯JSON格式（不要markdown包裹）：
{{
    "score": <0.0到1.0的浮点数>,
    "covered_points": <被覆盖的信息点数>,
    "total_points": <参考答案总信息点数>,
    "missing_info": ["<检索文档中缺失的关键信息1>", "..."],
    "reasoning": "<简要说明，中文>"
}}"""


def evaluate_context_recall(question: str, contexts: list[str], ground_truth: str) -> dict:
    """评估上下文召回率。"""
    docs_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {ctx[:800]}" for i, ctx in enumerate(contexts)
    )

    user_prompt = f"""用户问题: {question}

参考答案（Ground Truth）:
{ground_truth}

检索到的文档片段（共{len(contexts)}个）:
{docs_text}

请评估检索结果对参考答案的覆盖率。"""

    client = _get_client()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": RECALL_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


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
