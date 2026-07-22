"""DeepReason 检索与生成质量评估 — 用 DeepSeek API 直接评估 4 指标

- faithfulness（忠实度）: 答案是否基于检索上下文
- answer_relevancy（答案相关性）: 答案与问题的匹配度
- context_precision（上下文精确率）: 检索结果的信号噪声比
- context_recall（上下文召回率）: 检索是否覆盖了 ground truth 要点

使用方式:
    python eval/run_eval.py                    # 完整评估 (HyDE 开启)
    python eval/run_eval.py --no-hyde          # 禁用 HyDE 的对照组
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table

from config.settings import (
    EMBEDDING_MODEL_NAME,
    PARENT_CHUNKS_FILE,
    BM25_INDEX_DIR,
    FINAL_TOP_K,
)
from src.retrieval.bm25_store import BM25Store
from src.retrieval.reranker import Reranker
from src.retrieval.hyde import HyDE
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import (
    connect_milvus,
    disconnect_milvus,
    load_parents,
)
from src.utils.llm import call_llm, JUDGE_MODEL

console = Console()


# ============================================================================
# 数据加载
# ============================================================================

def load_questions(path: str) -> list[dict]:
    """加载评估问题集。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# 答案生成
# ============================================================================

ANSWER_SYSTEM_PROMPT = """你是一个AI技术知识问答助手。请基于以下检索到的上下文来回答用户的问题。

要求:
1. 严格基于提供的上下文来回答，不要编造信息
2. 如果上下文不足以回答某个部分，请明确说"基于现有资料无法确定"
3. 使用中文回答，专业术语可保留英文
4. 回答要条理清晰，适当分点说明"""


def generate_answer(query: str, contexts: list[str]) -> str:
    """用 DeepSeek API 基于检索到的上下文生成答案。"""
    context_text = "\n\n---\n\n".join(
        f"[来源 {i+1}] {ctx}" for i, ctx in enumerate(contexts)
    )
    user_prompt = f"""上下文资料:
{context_text}

问题: {query}

请基于以上上下文回答问题。"""

    return call_llm(JUDGE_MODEL, ANSWER_SYSTEM_PROMPT, user_prompt, temperature=0.3)


# ============================================================================
# 自定义评估指标（DeepSeek API 直接评估）
# ============================================================================

EVAL_SYSTEM_PROMPT = """你是一个严格的RAG系统评估专家。你需要对检索增强生成系统的输出进行质量评估。
请严格按照JSON格式输出评估结果，不要输出任何JSON之外的内容。"""


def evaluate_faithfulness(question: str, answer: str, contexts: list[str]) -> dict:
    """评估答案忠实度——答案是否基于检索上下文。"""
    context_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {ctx[:1000]}" for i, ctx in enumerate(contexts)
    )
    user_prompt = f"""评估任务: Faithfulness（忠实度）

请评估"生成的答案"是否完全基于"提供的上下文"。

评估步骤:
1. 逐句检查生成答案中的每个事实性断言
2. 判断每个断言是否能在上下文中找到直接依据
3. 标记在上下文中找不到依据的"幻觉"断言
4. 综合给出忠实度评分(0.0-1.0)

评分标准:
- 1.0: 所有断言都在上下文中有直接依据
- 0.7-0.9: 大部分有依据，少量合理推断
- 0.4-0.6: 约一半有依据，存在明显无依据陈述
- 0.1-0.3: 大部分无依据
- 0.0: 与上下文完全无关

---
问题: {question}

提供的上下文:
{context_text}

生成的答案:
{answer}
---

请输出纯JSON:
{{"score": <0.0到1.0>, "hallucination_count": <整数>, "total_claims": <整数>, "reasoning": "<1-2句中评语>"}}"""

    result_str = call_llm(JUDGE_MODEL, EVAL_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    # 清洗可能的 markdown 包裹
    result_str = result_str.strip()
    if result_str.startswith("```"):
        result_str = result_str.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(result_str)


def evaluate_answer_relevancy(question: str, answer: str) -> dict:
    """评估答案相关性——答案是否切题。"""
    user_prompt = f"""评估任务: Answer Relevancy（答案相关性）

请评估"生成的答案"与"用户问题"的匹配程度。

评估步骤:
1. 分析用户问题的核心意图
2. 检查答案是否直接回应了问题
3. 检查答案中是否有冗余或偏题内容
4. 综合给出相关性评分(0.0-1.0)

评分标准:
- 1.0: 精准回应所有要点
- 0.7-0.9: 回应了主要问题
- 0.4-0.6: 部分回应，有偏题
- 0.1-0.3: 大部分不相关
- 0.0: 完全答非所问

---
用户问题: {question}

生成的答案:
{answer}
---

请输出纯JSON:
{{"score": <0.0到1.0>, "covered_points": <覆盖的信息点数>, "total_points": <问题总信息点数>, "reasoning": "<1-2句中评语>"}}"""

    result_str = call_llm(JUDGE_MODEL, EVAL_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    result_str = result_str.strip()
    if result_str.startswith("```"):
        result_str = result_str.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(result_str)


def evaluate_context_precision(question: str, contexts: list[str]) -> dict:
    """评估上下文精确率——检索到的文档有多少真正相关。"""
    docs_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {ctx[:800]}" for i, ctx in enumerate(contexts)
    )
    user_prompt = f"""评估任务: Context Precision（上下文精确率）

请评估检索到的文档片段中有多少真正与用户问题相关。

评估步骤:
1. 分析用户问题需要什么信息
2. 逐个检查每个检索片段是否包含有用信息
3. 标记相关/不相关
4. 计算相关占比(0.0-1.0)

评分标准:
- 1.0: 所有片段都相关
- 0.7-0.9: 大部分相关，少量噪声
- 0.4-0.6: 约一半相关
- 0.1-0.3: 大部分不相关
- 0.0: 全部不相关

---
用户问题: {question}

检索到的片段（共{len(contexts)}个）:
{docs_text}
---

请输出纯JSON:
{{"score": <0.0到1.0>, "relevant_count": <相关片段数>, "total_documents": {len(contexts)}, "reasoning": "<1-2句中评语>"}}"""

    result_str = call_llm(JUDGE_MODEL, EVAL_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    result_str = result_str.strip()
    if result_str.startswith("```"):
        result_str = result_str.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(result_str)


def evaluate_context_recall(question: str, contexts: list[str], ground_truth: str) -> dict:
    """评估上下文召回率——检索是否覆盖了ground truth要点。"""
    docs_text = "\n\n---\n\n".join(
        f"[文档{i+1}] {ctx[:800]}" for i, ctx in enumerate(contexts)
    )
    user_prompt = f"""评估任务: Context Recall（上下文召回率）

请评估检索到的文档片段是否覆盖了参考答案中的关键信息点。

评估步骤:
1. 从参考答案中提取关键事实性信息点
2. 逐一检查每个信息点是否能在检索文档中找到
3. 列出检索文档中缺失的关键信息
4. 计算被覆盖的信息点占比(0.0-1.0)

评分标准:
- 1.0: 所有信息点都在检索文档中
- 0.7-0.9: 大部分被覆盖
- 0.4-0.6: 约一半被覆盖
- 0.1-0.3: 少量被覆盖
- 0.0: 完全不包含参考答案信息

---
用户问题: {question}

参考答案(Ground Truth):
{ground_truth}

检索到的片段（共{len(contexts)}个）:
{docs_text}
---

请输出纯JSON:
{{"score": <0.0到1.0>, "covered_points": <被覆盖信息点数>, "total_points": <参考答案总信息点数>, "missing_info": ["缺失的信息1", "缺失的信息2"], "reasoning": "<1-2句中评语>"}}"""

    result_str = call_llm(JUDGE_MODEL, EVAL_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    result_str = result_str.strip()
    if result_str.startswith("```"):
        result_str = result_str.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(result_str)


def evaluate_all(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    """对单个问题运行全部 4 个评估指标。"""
    return {
        "faithfulness": evaluate_faithfulness(question, answer, contexts),
        "answer_relevancy": evaluate_answer_relevancy(question, answer),
        "context_precision": evaluate_context_precision(question, contexts),
        "context_recall": evaluate_context_recall(question, contexts, ground_truth),
    }


# ============================================================================
# 检索统计
# ============================================================================

@dataclass
class RetrievalStats:
    """单次检索的详细统计。"""
    query: str = ""
    final_count: int = 0
    expansion_triggered: bool = False
    hyde_enabled: bool = True


def compute_retrieval_stats(
    hybrid_hits: list[dict],
    hyde_enabled: bool = True,
) -> RetrievalStats:
    """为单次检索计算自定义统计指标。"""
    stats = RetrievalStats(hyde_enabled=hyde_enabled)
    stats.final_count = len(hybrid_hits)
    for hit in hybrid_hits:
        if hit.get("child_count", 0) >= 2:
            stats.expansion_triggered = True
    return stats


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepReason 检索质量评估")
    parser.add_argument("--no-hyde", action="store_true", help="禁用 HyDE（对照组）")
    parser.add_argument(
        "--questions",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "questions.json"),
        help="评估问题集路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制评估问题数量（0=全部）",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    console.rule("[bold blue]DeepReason 检索质量评估 (4 指标)")
    console.print(f"Embedding 模型: {EMBEDDING_MODEL_NAME}")
    console.print(f"HyDE: {'[yellow]禁用[/]' if args.no_hyde else '[green]启用[/]'}")
    console.print(f"评估 LLM: {JUDGE_MODEL}")

    questions = load_questions(args.questions)
    if args.limit > 0:
        questions = questions[:args.limit]
    console.print(f"评估问题数: {len(questions)}")

    # 加载检索组件
    console.print("\n[dim]初始化检索组件...[/]")
    load_parents(PARENT_CHUNKS_FILE)
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    bm25 = BM25Store()
    bm25.load(BM25_INDEX_DIR)
    reranker = Reranker()
    hyde = HyDE() if not args.no_hyde else None
    connect_milvus()

    retriever = HybridRetriever(
        bm25_store=bm25,
        reranker=reranker,
        hyde=hyde,
        enable_hyde=not args.no_hyde,
    )

    # ------------------------------------------------------------------
    # 逐题评估
    # ------------------------------------------------------------------
    all_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }
    all_stats: list[RetrievalStats] = []
    per_question_results: list[dict] = []

    for i, q in enumerate(questions):
        console.print(f"\n[bold cyan]━━━ [{i+1}/{len(questions)}][/] [{q['difficulty']}] {q['question'][:55]}...")

        # 检索
        t0 = time.time()
        query_embedding = emb_model.encode(
            q["question"], normalize_embeddings=True
        ).astype(np.float32)
        hybrid_hits = retriever.search(
            query=q["question"],
            query_embedding=query_embedding,
            top_k=FINAL_TOP_K,
        )
        retrieval_time = time.time() - t0

        contexts = [hit["content"] for hit in hybrid_hits]

        # 生成答案
        t0 = time.time()
        answer = generate_answer(q["question"], contexts)
        gen_time = time.time() - t0

        # 评估 4 指标
        console.print("  [dim]⏳ DeepSeek 评估中...[/]", end="")
        eval_results = evaluate_all(q["question"], answer, contexts, q["ground_truth"])
        console.print(" 完成")

        # 统计
        stats = compute_retrieval_stats(hybrid_hits, hyde_enabled=not args.no_hyde)
        all_stats.append(stats)

        for metric in all_scores:
            score = eval_results[metric].get("score", 0)
            all_scores[metric].append(score)

        per_question_results.append({
            "id": q["id"],
            "question": q["question"],
            "difficulty": q["difficulty"],
            "answer": answer,
            "num_contexts": len(contexts),
            "expansion_triggered": stats.expansion_triggered,
            "scores": {
                k: {"score": eval_results[k].get("score"), **eval_results[k]}
                for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            },
        })

        # 简要输出
        scores_str = " | ".join(
            f"{k.split('_')[-1][:4]}={eval_results[k].get('score', 0):.2f}"
            for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        )
        console.print(
            f"  [dim]检索 {len(contexts)}条 ({retrieval_time:.1f}s) | 生成 ({gen_time:.1f}s)[/]"
        )
        console.print(f"  [bold]{scores_str}[/]")

    disconnect_milvus()

    # ------------------------------------------------------------------
    # 汇总输出
    # ------------------------------------------------------------------
    console.print()
    console.rule("[bold green]评估汇总")

    # 指标表
    metrics_table = Table(title="评估指标汇总", show_header=True)
    metrics_table.add_column("指标", style="cyan")
    metrics_table.add_column("中文名", style="dim")
    metrics_table.add_column("平均分", justify="right")
    metrics_table.add_column("最高", justify="right")
    metrics_table.add_column("最低", justify="right")
    metrics_table.add_column("解读")

    metric_labels = {
        "faithfulness": ("忠实度", "答案是否基于检索上下文，无幻觉"),
        "answer_relevancy": ("答案相关性", "答案与问题的匹配程度"),
        "context_precision": ("上下文精确率", "检索结果的信噪比"),
        "context_recall": ("上下文召回率", "检索是否覆盖全部必要信息"),
    }

    for key, (cn_name, desc) in metric_labels.items():
        scores = all_scores[key]
        avg_s = np.mean(scores) if scores else 0
        max_s = max(scores) if scores else 0
        min_s = min(scores) if scores else 0
        emoji = "[green]●[/]" if avg_s > 0.7 else ("[yellow]●[/]" if avg_s > 0.4 else "[red]●[/]")
        metrics_table.add_row(
            f"{emoji} {key}", cn_name,
            f"{avg_s:.4f}", f"{max_s:.4f}", f"{min_s:.4f}",
            desc,
        )

    console.print(metrics_table)

    # 自定义指标
    avg_hits = np.mean([s.final_count for s in all_stats])
    expansion_rate = np.mean([1 if s.expansion_triggered else 0 for s in all_stats])

    custom_table = Table(title="自定义检索指标", show_header=True)
    custom_table.add_column("指标", style="cyan")
    custom_table.add_column("值", justify="right")
    custom_table.add_column("说明")
    custom_table.add_row("平均召回数", f"{avg_hits:.1f}", "每次查询返回的 Parent Chunk 数")
    custom_table.add_row(
        "Child→Parent 扩展触发率", f"{expansion_rate:.0%}",
        "≥2个child来自同一parent时触发扩展"
    )
    custom_table.add_row("HyDE 状态", "禁用" if args.no_hyde else "启用", "")
    console.print(custom_table)

    # 逐题表
    detail_table = Table(title="逐题得分详情", show_header=True)
    detail_table.add_column("#", width=3)
    detail_table.add_column("问题", width=32)
    detail_table.add_column("难度", width=5)
    detail_table.add_column("Faith.", width=6, justify="right")
    detail_table.add_column("Relev.", width=6, justify="right")
    detail_table.add_column("C.Prec.", width=6, justify="right")
    detail_table.add_column("C.Rec.", width=6, justify="right")
    detail_table.add_column("Hits", width=4, justify="right")

    for i, q in enumerate(questions):
        detail_table.add_row(
            str(i + 1), q["question"][:30], q["difficulty"],
            f"{all_scores['faithfulness'][i]:.3f}",
            f"{all_scores['answer_relevancy'][i]:.3f}",
            f"{all_scores['context_precision'][i]:.3f}",
            f"{all_scores['context_recall'][i]:.3f}",
            str(all_stats[i].final_count),
        )

    console.print(detail_table)

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)

    hyde_tag = "no_hyde" if args.no_hyde else "with_hyde"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    result_path = os.path.join(
        os.path.dirname(__file__), "results", f"eval_{hyde_tag}_{timestamp}.json"
    )

    avg_scores = {
        f"avg_{k}": float(np.mean(v)) if v else 0
        for k, v in all_scores.items()
    }

    full_results = {
        "config": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "hyde_enabled": not args.no_hyde,
            "final_top_k": FINAL_TOP_K,
            "num_questions": len(questions),
            "eval_llm": JUDGE_MODEL,
        },
        "summary": avg_scores,
        "custom_metrics": {
            "avg_recall_count": float(avg_hits),
            "child_parent_expansion_rate": float(expansion_rate),
        },
        "per_question": per_question_results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    # CSV
    csv_path = result_path.replace(".json", ".csv")
    csv_rows = []
    for item in per_question_results:
        csv_rows.append({
            "ID": item["id"],
            "问题": item["question"],
            "难度": item["difficulty"],
            "Faithfulness": item["scores"]["faithfulness"]["score"],
            "Answer Relevancy": item["scores"]["answer_relevancy"]["score"],
            "Context Precision": item["scores"]["context_precision"]["score"],
            "Context Recall": item["scores"]["context_recall"]["score"],
            "检索命中数": item["num_contexts"],
            "Parent扩展触发": item["expansion_triggered"],
        })
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    console.print(f"\n[dim]完整结果已保存到: {result_path}[/]")
    console.print(f"[dim]CSV 已保存到: {csv_path}[/]")

    # 最终判定
    console.print()
    console.rule("[bold blue]判定")
    total_avg = np.mean([float(np.mean(v)) for v in all_scores.values() if v])
    console.print(f"[bold]4指标平均分: {total_avg:.4f}[/]")
    if total_avg > 0.7:
        console.print("[green]✅ 检索质量良好，可以进入 Step 3 LangGraph 搭建[/]")
    elif total_avg > 0.5:
        console.print("[yellow]⚠ 检索质量中等，建议优化后再进入 Step 3[/]")
    else:
        console.print("[red]❌ 检索质量偏低，请检查检索链路配置[/]")


if __name__ == "__main__":
    main()
