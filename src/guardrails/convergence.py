"""护栏模块 — 收敛检测

审查-修订循环目前由轮数上限驱动（最多 MAX_REVIEW_ROUNDS 轮）。
收敛检测让循环由「答案是否还在实质变化」驱动：当 Reviser 的修订
与上一版答案高度相似（未实质修改）时，判定已收敛，提前终止——

这是防过度修正（q001 型：完美草稿被越改越坏）的关键护栏：
修订只动了只言片语 → 继续循环收益趋近于零，风险却仍在累积。
"""

from difflib import SequenceMatcher

from config.settings import CONVERGENCE_SIMILARITY_THRESHOLD


def answer_similarity(prev_answer: str, curr_answer: str) -> float:
    """计算两个答案文本的相似度（difflib 字符级 Ratio，0.0-1.0）。

    Args:
        prev_answer: 修订前的答案（draft 或上一版 refined）。
        curr_answer: 修订后的答案。

    Returns:
        相似度分数。1.0 = 完全一致。
    """
    if not prev_answer or not curr_answer:
        return 0.0
    return SequenceMatcher(None, prev_answer, curr_answer).ratio()


def check_answer_convergence(
    prev_answer: str,
    curr_answer: str,
    threshold: float = CONVERGENCE_SIMILARITY_THRESHOLD,
) -> dict:
    """判断本次修订是否构成实质变化。

    Returns:
        {"similarity": float, "converged": bool}
        converged=True 表示修订与上一版高度相似（≥阈值），
        应提前终止审查-修订循环。
    """
    sim = answer_similarity(prev_answer, curr_answer)
    return {"similarity": sim, "converged": sim >= threshold}
