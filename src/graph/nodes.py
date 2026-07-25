"""图节点函数 — 3-Agent 审查-修订引擎

7 个节点：
  retrieve → plan → multi_hop_retrieve → generator
    → critic → reviser → critic (loop) → finalize

节点通过 contextvars 访问共享的 HybridRetriever 和 Embedding 模型。
"""

import contextvars
import time

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from config.settings import FINAL_TOP_K, HYDE_ENABLED
from src.agents.planner import call_planner
from src.agents.generator import call_generator
from src.agents.critic import call_critic, call_critic_verify
from src.agents.reviser import call_reviser

console = Console()

# ============================================================================
# Context Variables
# ============================================================================

_retriever_ctx: contextvars.ContextVar = contextvars.ContextVar("retriever", default=None)
_embed_model_ctx: contextvars.ContextVar = contextvars.ContextVar("embed_model", default=None)


def set_retriever_context(retriever, embed_model):
    _retriever_ctx.set(retriever)
    _embed_model_ctx.set(embed_model)


# ============================================================================
# 工具函数
# ============================================================================

def _truncate(text: str, max_chars: int = 600) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... [已截断]"


def _verdict_style(verdict: str) -> str:
    styles = {
        "accept": "[bold green]✅ ACCEPT[/bold green]",
        "revise": "[bold yellow]🔧 REVISE[/bold yellow]",
        "reject": "[bold red]❌ REJECT[/bold red]",
    }
    return styles.get(verdict, f"[dim]{verdict}[/dim]")


# ============================================================================
# Node: retrieve — 初始检索
# ============================================================================

def retrieve_node(state: dict) -> dict:
    retriever = _retriever_ctx.get()
    embed_model = _embed_model_ctx.get()
    query = state["query"]

    console.print(Rule("🔍 检索阶段", style="blue"))

    query_emb = embed_model.encode(query, normalize_embeddings=True).astype(np.float32)

    docs = retriever.search(query=query, query_embedding=query_emb, top_k=FINAL_TOP_K)

    title_lines = []
    for i, d in enumerate(docs):
        t = d.get("title", "未知")
        r = d.get("score", 0)
        title_lines.append(f"  [{i+1}] {t} (相关度: {r:.4f})")
    console.print(f"检索到 [bold]{len(docs)}[/bold] 篇文档:")
    console.print("\n".join(title_lines))

    return {"retrieved_docs": docs, "retrieval_hops": 1}


# ============================================================================
# Node: plan — 问题拆解
# ============================================================================

def plan_node(state: dict) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs", [])

    doc_summaries = [
        f"{d.get('title', '未知')}: {d.get('content', '')[:200].strip()}..."
        for d in docs[:5]
    ]

    console.print(Rule("📋 规划阶段", style="blue"))
    console.print(f"问题: [bold]{query}[/bold]")

    plan_result = call_planner(query, doc_summaries)
    complexity = plan_result["complexity"]
    sub_questions = plan_result["sub_questions"]

    if complexity == "simple":
        console.print("复杂度: [green]简单[/green] — 无需拆解子问题")
    else:
        console.print(f"复杂度: [yellow]多跳推理[/yellow] — 拆解为 {len(sub_questions)} 个子问题:")
        for i, sq in enumerate(sub_questions):
            console.print(f"  [{i+1}] {sq}")

    return {"complexity": complexity, "sub_questions": sub_questions}


# ============================================================================
# Node: multi_hop_retrieve — 子问题检索
# ============================================================================

def multi_hop_retrieve_node(state: dict) -> dict:
    retriever = _retriever_ctx.get()
    embed_model = _embed_model_ctx.get()

    sub_questions = state.get("sub_questions", [])
    existing_docs = state.get("retrieved_docs", [])
    previous_hops = state.get("retrieval_hops", 1)

    sub_docs: dict[str, list[dict]] = {}
    all_new_docs: list[dict] = []
    seen_ids = {d.get("chunk_id") for d in existing_docs}

    prev_context = "\n\n".join(d.get("content", "")[:500] for d in existing_docs[:3])

    console.print(Rule("🔄 多跳检索", style="blue"))

    for sq in sub_questions:
        console.print(f"  [bold]子问题[/bold]: {sq}")
        query_emb = embed_model.encode(sq, normalize_embeddings=True).astype(np.float32)

        results = retriever.search_multi_hop(
            query=sq,
            query_embedding=query_emb,
            previous_context=prev_context,
            top_k=max(3, FINAL_TOP_K // 2),
        )

        sub_docs[sq] = results
        new_count = 0
        for doc in results:
            cid = doc.get("chunk_id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_new_docs.append(doc)
                new_count += 1
        console.print(f"    → 检索到 {len(results)} 篇，新增 {new_count} 篇")

    merged_docs = existing_docs + all_new_docs
    console.print(f"合并后共 [bold]{len(merged_docs)}[/bold] 篇文档")

    return {
        "sub_question_docs": sub_docs,
        "retrieved_docs": merged_docs,
        "retrieval_hops": previous_hops + 1,
    }


# ============================================================================
# Node: generator — 生成答案草稿
# ============================================================================

def generator_node(state: dict) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs", [])

    console.print(Rule("📝 生成答案草稿", style="blue"))

    draft = call_generator(query, docs)

    console.print(Panel(_truncate(draft, 500), title="📄 草稿"))

    return {
        "draft_answer": draft,
        "review_round": 0,
        "revision_round": 0,
    }


# ============================================================================
# Node: critic — 审查答案
# ============================================================================

def critic_node(state: dict) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    current_answer = state.get("refined_answer") or state.get("draft_answer", "")
    review_round = state.get("review_round", 0) + 1

    console.print(Rule(f"🔍 审查 R{review_round}", style="yellow"))

    ruling = call_critic(
        query=query,
        answer=current_answer,
        docs=docs,
        review_round=review_round,
    )

    verdict = ruling.get("verdict", "unknown")
    confidence = ruling.get("confidence", 0.0)
    issues = ruling.get("issues", [])
    improvements = ruling.get("improvements", "")

    console.print(f"  ⚖️  [bold]Critic[/bold]: {_verdict_style(verdict)}  {confidence:.0%}")
    if issues:
        sevs = {"critical": 0, "minor": 0}
        for iss in issues:
            sevs[iss.get("severity", "minor")] += 1
        console.print(f"    issues: {len(issues)} (critical:{sevs['critical']} minor:{sevs['minor']})")
    if improvements and verdict != "accept":
        console.print(f"    [dim]{improvements[:200]}[/dim]")

    review_entry = {
        "round": review_round,
        "verdict": verdict,
        "confidence": confidence,
        "issues_count": len(issues),
        "timestamp": time.time(),
    }

    # 筛选高置信度 issues（conf>0.8），供 Reviser 定向修改和 Verifier 定向检查
    high_conf_issues = [i for i in issues if i.get("confidence", 0) >= 0.8]
    if high_conf_issues and verdict != "accept":
        console.print(f"    [cyan]→ Reviser 将处理 {len(high_conf_issues)} 个高置信度 issues[/]")
    elif issues and not high_conf_issues:
        console.print("    [dim]→ 无高置信度 issue，跳过修订[/]")

    return {
        "critic_ruling": ruling,
        "review_round": review_round,
        "previous_issues": high_conf_issues,
        "review_history": [review_entry],
    }


# ============================================================================
# Node: reviser — 根据审查意见修订答案
# ============================================================================

def reviser_node(state: dict) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    current_answer = state.get("refined_answer") or state.get("draft_answer", "")
    # 只修 Critic(full) 筛出的高置信度 issues（conf>0.8）——低置信度的边角料不改，避免过修正
    issues = state.get("previous_issues", [])
    if not issues:
        # 兜底：如果 previous_issues 为空（首轮或异常），用 critic_ruling 的全部 issues 并过滤
        issues = [i for i in state.get("critic_ruling", {}).get("issues", []) if i.get("confidence", 0) >= 0.8]
    improvements = state.get("critic_ruling", {}).get("improvements", "")
    revision_round = state.get("revision_round", 0) + 1

    console.print(Rule(f"📝 修订 R{revision_round}", style="yellow"))
    console.print(f"  [cyan]处理 {len(issues)} 个高置信度 issue[/]")

    # 将 critic 的结构化 issues 格式化为可读的审稿意见
    issues_text = ""
    for i, iss in enumerate(issues):
        sev = "严重" if iss.get("severity") == "critical" else "次要"
        issues_text += f"{i+1}. [{sev}] {iss.get('claim', '')}\n"
        fix = iss.get("fix", "")
        if fix:
            issues_text += f"   修改建议: {fix}\n"

    critique = f"""## 主编审稿意见（仅列出高置信度问题）

### 需要修改的问题
{issues_text if issues_text else "（无具体问题）"}

### 整体修改方向
{improvements}

请逐一修改以上问题。先处理标记为"严重"的条目。
仅修改列出的问题，不要顺手修改其他段落。"""

    refined = call_reviser(
        query=query,
        current_answer=current_answer,
        critique=critique,
        docs=docs,
    )

    console.print(Panel(_truncate(refined, 500), title="📝 修订后答案"))

    revision_entry = {
        "round": revision_round,
        "issues_count": len(issues),
        "before": current_answer[:200],
        "after": refined[:200],
    }

    return {
        "refined_answer": refined,
        "revision_round": revision_round,
        "revision_history": [revision_entry],
    }


# ============================================================================
# Node: verify — 定向检查 issues 是否被修完
# ============================================================================

def verify_node(state: dict) -> dict:
    """逐条检查上一轮 Critic 的高置信度 issues 是否已被 Reviser 解决。"""
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    current_answer = state.get("refined_answer") or state.get("draft_answer", "")
    previous_issues = state.get("previous_issues", [])

    console.print(Rule("🔬 定向验证", style="cyan"))

    result = call_critic_verify(
        query=query,
        answer=current_answer,
        previous_issues=previous_issues,
        docs=docs,
    )

    all_resolved = result.get("all_resolved", False)
    details = result.get("details", [])

    resolved_count = sum(1 for d in details if d.get("resolved") == True)
    console.print(f"  {resolved_count}/{len(previous_issues)} issues 已解决  {'✅ 全部通过' if all_resolved else '🔧 需要继续修改'}")

    return {
        "verify_result": result,
    }


# ============================================================================
# Node: finalize — 输出最终结果
# ============================================================================

def finalize_node(state: dict) -> dict:
    answer = state.get("refined_answer") or state.get("draft_answer", "")
    ruling = state.get("critic_ruling", {})
    docs = state.get("retrieved_docs", [])
    review_round = state.get("review_round", 0)
    revision_round = state.get("revision_round", 0)

    verdict = ruling.get("verdict", "unknown")
    confidence = ruling.get("confidence", 0.5)

    console.print(Rule("🏁 最终结果", style="bold green"))
    console.print(
        f"  {_verdict_style(verdict)}  置信度 [bold]{confidence:.0%}[/bold]"
        f"  审查{review_round}轮  修订{revision_round}轮"
    )
    console.print(Panel(answer, title="📋 最终答案", border_style="green"))

    return {
        "final_answer": answer,
        "confidence": confidence,
    }
