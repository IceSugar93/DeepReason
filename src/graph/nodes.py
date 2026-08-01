"""图节点函数 — 3-Agent 审查-修订引擎

7 个节点：
  retrieve → plan → multi_hop_retrieve → generator
    → critic → reviser → critic (loop) → finalize

节点通过 contextvars 访问共享的 HybridRetriever 和 Embedding 模型。
"""

import contextvars
import json
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
from src.guardrails.convergence import check_answer_convergence
from src.guardrails.safety import build_answer_annotations
from src.mcp_server.runtime import register_retriever

console = Console()

# ============================================================================
# Context Variables
# ============================================================================

_retriever_ctx: contextvars.ContextVar = contextvars.ContextVar("retriever", default=None)
_embed_model_ctx: contextvars.ContextVar = contextvars.ContextVar("embed_model", default=None)


def set_retriever_context(retriever, embed_model):
    _retriever_ctx.set(retriever)
    _embed_model_ctx.set(embed_model)
    # 同步注册给 mcp_server 工具后端（Critic 工具核查等进程内调用使用）
    register_retriever(retriever)


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
        console.print(f"    审查意见（{len(issues)} 条）:")
        for idx, iss in enumerate(issues, 1):
            sev = "严重" if iss.get("severity") == "critical" else "次要"
            iss_conf = iss.get("confidence", 0)
            conf_str = f"{float(iss_conf):.2f}" if isinstance(iss_conf, (int, float)) else str(iss_conf)
            console.print(
                f"      {idx}. [{sev}][{iss.get('status', '?')}][conf {conf_str}] {iss.get('claim', '')}"
            )
            evidence = iss.get("evidence", "")
            if evidence:
                console.print(f"         [dim]证据: {evidence}[/]")
            fix = iss.get("fix", "")
            if fix:
                console.print(f"         [dim]→ 修改建议: {fix}[/]")
    if improvements:
        console.print(f"    [dim]整体方向: {improvements}[/]")
    reasoning = ruling.get("reasoning", "")
    if reasoning:
        console.print(f"    [dim]判定理由: {reasoning}[/]")

    # ── 工具核查结果展示 + 命中文档并入检索集 ──
    tool_verifications = ruling.get("tool_verifications", [])
    updated_docs = docs
    if tool_verifications:
        intercepted = sum(1 for v in tool_verifications if v.get("verdict") == "supported")
        console.print(
            f"    [cyan]🔧 工具核查 {len(tool_verifications)} 条:"
            f"{intercepted} 条改判 supported（已拦截）[/]"
        )
        for idx, v in enumerate(tool_verifications, 1):
            v_style = "[green]supported[/]" if v.get("verdict") == "supported" else "[red]unsupported[/]"
            console.print(
                f"      {idx}. {v_style} {v.get('claim', '')}"
            )
            if v.get("reason"):
                console.print(f"         [dim]理由: {v['reason']}[/]")
            fetched_ids = v.get("fetched_chunk_ids", [])
            if fetched_ids:
                console.print(f"         [dim]命中文档: {', '.join(fetched_ids)}[/]")
            for tc in v.get("tool_trace", []):
                console.print(
                    f"         [dim]  · {tc.get('name')}({tc.get('args', {})}) → {tc.get('hits', 0)} 条命中[/]"
                )
        from src.retrieval.vector_store import lookup_parents

        fetched_ids = []
        for v in tool_verifications:
            fetched_ids.extend(v.get("fetched_chunk_ids", []))
        existing_ids = {d.get("chunk_id") for d in docs}
        new_docs = [
            d for d in lookup_parents(fetched_ids)
            if d.get("chunk_id") not in existing_ids
        ]
        if new_docs:
            # 工具取回的文档并入检索集——它们已成为裁决依据，必须流进
            # internal_contexts，否则评估侧会把有据断言误判为幻觉
            updated_docs = docs + new_docs
            console.print(f"    [cyan]→ {len(new_docs)} 篇工具命中文档并入检索集[/]")

    # ── 完整输出展示：模型原始 JSON + 工具核查后的最终裁决全量 ──
    raw_output = ruling.get("raw_output", "")
    if raw_output:
        console.print(Panel(
            raw_output,
            title=f"📜 Critic 原始输出 R{review_round}（模型原文）",
            border_style="cyan",
        ))
    final_ruling = {k: v for k, v in ruling.items() if k != "raw_output"}
    console.print(Panel(
        json.dumps(final_ruling, ensure_ascii=False, indent=2),
        title=f"🔬 Critic 完整裁决 R{review_round}（含工具核查后）",
        border_style="magenta",
    ))

    review_entry = {
        "round": review_round,
        "verdict": verdict,
        "confidence": confidence,
        "issues_count": len(issues),
        # 完整保留本轮审查意见，供评估脚本展示逐轮 Critic 观点
        "issues": issues,
        "improvements": improvements,
        "reasoning": ruling.get("reasoning", ""),
        "tool_verifications": tool_verifications,
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
        "retrieved_docs": updated_docs,
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
        issues_text += f"{i+1}. [{sev}][{iss.get('status', '?')}] {iss.get('claim', '')}\n"
        evidence = iss.get("evidence", "")
        if evidence:
            issues_text += f"   文献依据: {evidence}\n"
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

    # ── Guardrails: 收敛检测 — 修订与上一版高度相似 → 标记收敛，路由层据此提前终止 ──
    conv = check_answer_convergence(current_answer, refined)
    if conv["converged"]:
        console.print(
            f"    [yellow]🧿 收敛检测: 修订与上一版相似度 {conv['similarity']:.0%}，"
            "答案未实质变化，判定收敛，提前终止[/]"
        )

    console.print(Panel(_truncate(refined, 500), title="📝 修订后答案"))

    revision_entry = {
        "round": revision_round,
        "issues_count": len(issues),
        "before": current_answer[:200],
        "after": refined[:200],
        "similarity": conv["similarity"],
        "converged": conv["converged"],
    }

    return {
        "refined_answer": refined,
        "revision_round": revision_round,
        "revision_history": [revision_entry],
        "converged": conv["converged"],
        "answer_similarity": conv["similarity"],
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
    query = state.get("query", "")
    draft_answer = state.get("draft_answer", "")
    answer = state.get("refined_answer") or draft_answer
    ruling = state.get("critic_ruling", {})
    review_round = state.get("review_round", 0)
    revision_round = state.get("revision_round", 0)

    verdict = ruling.get("verdict", "unknown")
    confidence = ruling.get("confidence", 0.5)

    # ── Guardrails: reject 回退初稿 — 裁决为 reject 时修订版被认为
    #    风险高于初稿，退回 Generator 初稿作为最终答案 ──
    if verdict == "reject" and draft_answer and answer != draft_answer:
        console.print(
            f"    [red]❌ reject 回退: 修订版被认为不可信，最终答案退回初稿[/]"
        )
        answer = draft_answer

    # ── Guardrails: 风险/低置信度标注 — 只写入元数据，不污染 answer 文本 ──
    annotations = build_answer_annotations(query, confidence)
    if annotations["risk_warning"]:
        console.print(f"    [yellow]⚠️  {annotations['risk_warning']}[/]")
    if annotations["uncertainty_note"]:
        console.print(f"    [yellow]⚠️  {annotations['uncertainty_note']}[/]")

    console.print(Rule("🏁 最终结果", style="bold green"))
    console.print(
        f"  {_verdict_style(verdict)}  置信度 [bold]{confidence:.0%}[/bold]"
        f"  审查{review_round}轮  修订{revision_round}轮"
    )
    console.print(Panel(answer, title="📋 最终答案", border_style="green"))

    return {
        "final_answer": answer,
        "confidence": confidence,
        "answer_annotations": annotations,
    }
