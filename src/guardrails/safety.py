"""护栏模块 — 风险标注与低置信度标注

对高风险领域（法律/医疗/金融）的查询在最终答案附加人工复核提示；
对低置信度的最终答案附加不确定性标注。标注只写入元数据字段，
不污染 answer 文本本身（answer 会进入评估管线）。
"""

from config.settings import LOW_CONFIDENCE_ANNOTATION_THRESHOLD

# 风险领域关键词（面向 AI/技术知识库场景，只覆盖可能出格的三类）
RISK_KEYWORDS = {
    "legal": ["法律", "法规", "诉讼", "合规", "违法", "责任", "合同"],
    "medical": ["医疗", "诊断", "治疗", "药物", "副作用", "症状", "疾病"],
    "financial": ["投资", "理财", "收益", "风险", "金融", "税务", "股票"],
}


def detect_risk(query: str) -> list[str]:
    """检测用户查询涉及的风险领域。

    Args:
        query: 用户原始问题。

    Returns:
        命中的风险领域标签列表，如 ["legal", "medical"]；无命中返回 []。
    """
    risks = []
    for domain, keywords in RISK_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            risks.append(domain)
    return risks


def build_answer_annotations(
    query: str,
    confidence: float,
    low_confidence_threshold: float = LOW_CONFIDENCE_ANNOTATION_THRESHOLD,
) -> dict:
    """为最终答案生成标注元数据。

    Returns:
        {"risks": [...], "risk_warning": str|"",
         "uncertainty_note": str|"", "requires_human_review": bool}
    """
    risks = detect_risk(query)

    risk_warning = ""
    if risks:
        risk_warning = f"该问题涉及{'/'.join(risks)}领域，AI 生成内容仅供参考，建议人工复核"

    uncertainty_note = ""
    if confidence < low_confidence_threshold:
        uncertainty_note = "此结论置信度较低（{:.0%}），建议结合原始文献进一步验证".format(confidence)

    return {
        "risks": risks,
        "risk_warning": risk_warning,
        "uncertainty_note": uncertainty_note,
        "requires_human_review": bool(risk_warning),
    }
