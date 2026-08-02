/* DeepReason Web UI — 前端逻辑 */
"use strict";

const $ = (s) => document.querySelector(s);

/* ═══════════════ 工具 ═══════════════ */

function esc(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pct(x, digits = 1) {
  return x == null ? "—" : (x * 100).toFixed(digits) + "%";
}

const VERDICT_CN = { accept: "accept 通过", revise: "revise 修订", reject: "reject 拒绝" };
const STATUS_CN = {
  unsupported: "无据", contradicted: "矛盾", missing: "遗漏",
  supported: "有据", resolved: "已解决", unresolved: "未解决",
};

/* ═══════════════ Tab 切换 ═══════════════ */

$("#tab-demo").onclick = () => switchTab("demo");
$("#tab-eval").onclick = () => switchTab("eval");

function switchTab(name) {
  $("#tab-demo").classList.toggle("active", name === "demo");
  $("#tab-eval").classList.toggle("active", name === "eval");
  $("#panel-demo").classList.toggle("active", name === "demo");
  $("#panel-eval").classList.toggle("active", name === "eval");
  if (name === "eval" && !$("#eval-select").options.length) loadRuns();
}

/* ═══════════════ 引擎状态 ═══════════════ */

async function checkHealth() {
  const dot = $("#status-dot"), txt = $("#status-text");
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    if (h.ready) {
      dot.className = "dot green"; txt.textContent = `引擎就绪 · ${h.generator_model} · Top-K ${h.top_k}`;
    } else {
      dot.className = "dot red"; txt.textContent = `引擎未就绪: ${h.error || "初始化失败"}`;
    }
  } catch (e) {
    dot.className = "dot red"; txt.textContent = "无法连接后端";
  }
}

/* ═══════════════ 推理演示 ═══════════════ */

const NODE_UI = {
  retrieve:           { icon: "📚", name: "检索 retrieve" },
  plan:               { icon: "🧭", name: "规划 plan" },
  multi_hop_retrieve: { icon: "🔎", name: "多跳检索 multi_hop" },
  generator:          { icon: "✍️", name: "Generator 草稿生成" },
  critic:             { icon: "🧐", name: "Critic 审查与裁决" },
  reviser:            { icon: "🔧", name: "Reviser 修订" },
  verify:             { icon: "✅", name: "Verify 定向验证" },
  finalize:           { icon: "🏁", name: "Finalize 输出" },
};

let running = false;
let abortCtrl = null;

$("#btn-run").onclick = () => runDemo();
$("#btn-stop").onclick = () => { if (abortCtrl) abortCtrl.abort(); };
document.querySelectorAll(".examples a").forEach((a) => {
  a.onclick = () => { $("#query-input").value = a.dataset.q; runDemo(); };
});
$("#query-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runDemo();
});

async function runDemo() {
  const query = $("#query-input").value.trim();
  if (!query) { alert("请输入问题"); return; }
  if (running) return;

  running = true; abortCtrl = new AbortController();
  $("#btn-run").disabled = true; $("#btn-stop").disabled = false;
  $("#timeline").innerHTML = "";
  $("#result").classList.add("hidden");

  const url = "/api/stream?query=" + encodeURIComponent(query);
  try {
    const res = await fetch(url, { signal: abortCtrl.signal });
    if (!res.ok) { const e = await res.json().catch(() => ({})); showError(e.error || res.statusText); return; }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const line = chunk.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const evt = JSON.parse(line.slice(6));
        if (evt.type === "node") renderNode(evt.node, evt.data);
        else if (evt.type === "done") renderResult(evt.result);
        else if (evt.type === "error") showError(evt.message);
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") showError(String(e.message || e));
  } finally {
    running = false; abortCtrl = null;
    $("#btn-run").disabled = false; $("#btn-stop").disabled = true;
  }
}

function showError(msg) {
  const card = document.createElement("div");
  card.className = "node-card";
  card.innerHTML = `<div class="node-head"><span class="node-icon">⛔</span><span class="node-name">错误</span></div>
    <div class="node-body" style="color:var(--red)">${esc(msg)}</div>`;
  $("#timeline").appendChild(card);
}

function addNode(node, bodyHtml, meta) {
  const ui = NODE_UI[node] || { icon: "⚙️", name: node };
  const card = document.createElement("div");
  card.className = "node-card";
  card.innerHTML = `
    <div class="node-head">
      <span class="node-icon">${ui.icon}</span>
      <span class="node-name">${esc(ui.name)}</span>
      <span class="node-meta">${meta || ""}</span>
    </div>
    <div class="node-body">${bodyHtml}</div>`;
  $("#timeline").appendChild(card);
  card.scrollIntoView({ block: "nearest" });
  return card;
}

function docsHtml(docs) {
  if (!docs || !docs.length) return '<span class="muted">无</span>';
  return `<div class="docs">${docs.map((d) =>
    `<span class="doc-chip" title="${esc(d.chunk_id)}">${esc(d.title || d.chunk_id)}<span class="score">${d.score != null ? d.score.toFixed(3) : ""}</span></span>`
  ).join("")}</div>`;
}

function renderNode(node, data) {
  switch (node) {
    case "retrieve":
      addNode(node, `命中 <b>${data.retrieved_docs ? data.retrieved_docs.length : 0}</b> 篇文档（跳数 ${data.retrieval_hops || 1}）${docsHtml(data.retrieved_docs)}`);
      break;

    case "plan":
      const isMulti = data.complexity === "multi_hop";
      addNode(node,
        `复杂度: <span class="badge ${isMulti ? "revise" : "accept"}">${esc(data.complexity || "simple")}</span>` +
        (data.sub_questions && data.sub_questions.length
          ? `<div style="margin-top:4px">子问题：${data.sub_questions.map(esc).map((s) => `<div style="color:var(--cyan);font-size:12.5px">• ${s}</div>`).join("")}</div>`
          : ""));
      break;

    case "multi_hop_retrieve":
      const subDocs = data.sub_question_docs || {};
      addNode(node,
        Object.entries(subDocs).map(([q, docs]) =>
          `<div style="margin-bottom:6px"><div style="color:var(--cyan);font-size:12.5px">▸ ${esc(q)}</div>${docsHtml(docs)}</div>`
        ).join("") ||
        '<span class="muted">无子问题文档</span>');
      break;

    case "generator":
      addNode(node, `<div class="answer-text">${esc(data.draft_answer || "")}</div>`);
      break;

    case "critic": {
      const ruling = data.critic_ruling || {};
      const verdict = ruling.verdict || "?";
      const meta = `轮次 #${data.review_round ?? 1}`;
      let html = `
        <span class="badge ${verdict}">${VERDICT_CN[verdict] || verdict}</span>
        <span class="badge conf">置信度 ${pct(ruling.confidence)}</span>
        ${ruling.issues ? `<span class="badge dim">${ruling.issues.length} 条 issue</span>` : ""}`;
      if (ruling.issues && ruling.issues.length) {
        html += `<div style="margin-top:6px">${ruling.issues.map((i) => `
          <div class="issue">
            <div class="issue-claim">${esc(i.claim || "")}
              <span class="badge ${i.status === "contradicted" ? "reject" : "revise"}">${STATUS_CN[i.status] || esc(i.status)}</span>
              ${i.confidence != null ? `<span class="badge conf">${pct(i.confidence, 0)}</span>` : ""}
              ${i.severity ? `<span class="badge dim">${esc(i.severity)}</span>` : ""}
            </div>
            ${i.evidence ? `<div class="issue-ev">依据: <em>${esc(i.evidence)}</em></div>` : ""}
            ${i.fix ? `<div class="issue-ev">建议: ${esc(i.fix)}</div>` : ""}
          </div>`).join("")}</div>`;
      }
      if (ruling.tool_verifications && ruling.tool_verifications.length) {
        html += `<details class="raw"><summary>🔧 工具核查 ${ruling.tool_verifications.length} 条</summary>
          ${ruling.tool_verifications.map((v) =>
            `<div class="issue"><span class="badge ${v.verdict === "supported" ? "accept" : "reject"}">${esc(v.verdict || "?")}</span> ${esc(v.claim || "")}
             ${v.fetched_chunk_ids && v.fetched_chunk_ids.length ? `<div class="issue-ev">命中: ${v.fetched_chunk_ids.slice(0, 5).map(esc).join(", ")}</div>` : ""}
            </div>`).join("")}</details>`;
      }
      if (ruling.reasoning) {
        html += `<details class="raw"><summary>🧠 裁决推理</summary><pre class="raw-output">${esc(ruling.reasoning)}</pre></details>`;
      }
      addNode(node, html, meta);
      break;
    }

    case "reviser": {
      const metaParts = [];
      if (data.answer_similarity != null) metaParts.push(`相似度 ${pct(data.answer_similarity, 0)}`);
      if (data.converged) metaParts.push("已收敛");
      metaParts.push(`修订 #${data.revision_round ?? 1}`);
      addNode(node, `<div class="answer-text">${esc(data.refined_answer || "")}</div>`, metaParts.join(" · "));
      break;
    }

    case "verify": {
      const v = data.verify_result || {};
      const meta = v.all_resolved ? "全部解决" : "存在未解决";
      const badge = v.all_resolved ? '<span class="badge accept">✅ 全部解决</span>' : '<span class="badge reject">⚠ 存在未解决</span>';
      let html = badge;
      if (v.details && v.details.length) {
        html += `<div style="margin-top:6px">${v.details.map((d) => `
          <div class="issue"><span class="badge ${d.resolved ? "accept" : "reject"}">${d.resolved ? "已解决" : "未解决"}</span>
          issue #${d.issue_index ?? "?"} ${d.reason ? `<span class="issue-ev">${esc(d.reason)}</span>` : ""}</div>`).join("")}</div>`;
      }
      addNode(node, html, meta);
      break;
    }

    case "finalize":
      addNode(node,
        (data.answer_annotations && (data.answer_annotations.risk_annotations?.length || data.answer_annotations.low_confidence))
          ? `<span class="badge dim">已附加注解</span>` : '<span class="muted">无注解</span>',
        data.converged ? "收敛提前终止" : "");
      break;

    default:
      addNode(node, `<pre class="raw-output">${esc(JSON.stringify(data, null, 2))}</pre>`);
  }
}

function renderResult(r) {
  const rs = r.review_summary || {};
  const badges = [];
  badges.push(`<span class="badge ${r.verdict}">裁决: ${VERDICT_CN[r.verdict] || r.verdict}</span>`);
  badges.push(`<span class="badge conf">置信度 ${pct(r.confidence)}</span>`);
  badges.push(`<span class="badge dim">审查 ${rs.total_rounds || 0} 轮</span>`);
  badges.push(`<span class="badge dim">修订 ${rs.revision_rounds || 0} 轮</span>`);
  badges.push(`<span class="badge dim">检索 ${(r.retrieval_stats && r.retrieval_stats.hops) || 0} 跳</span>`);
  if (rs.converged) badges.push(`<span class="badge dim">已收敛 (${pct(rs.answer_similarity, 0)})</span>`);
  if (rs.issues_found) badges.push(`<span class="badge revise">发现 ${rs.issues_found} 条 issue</span>`);

  $("#result-badges").innerHTML = badges.join("");
  $("#result-answer").textContent = r.answer || "";

  const ann = r.answer_annotations || {};
  const annParts = [];
  if (ann.low_confidence) annParts.push(`⚠ 低置信度提示已附加`);
  if (ann.risk_annotations && ann.risk_annotations.length) annParts.push(`⚠ 风险领域: ${ann.risk_annotations.join(", ")}`);
  $("#result-annotations").innerHTML = annParts.map((t) => `<div>${esc(t)}</div>`).join("");

  const ctxs = r.internal_contexts || [];
  $("#result-contexts").innerHTML = ctxs.length
    ? ctxs.map((c, i) => `<div class="src-item"><b>[来源 ${i + 1}]</b> ${esc(c)}</div>`).join("")
    : '<span class="muted">无</span>';

  $("#result").classList.remove("hidden");
  $("#result").scrollIntoView({ block: "nearest" });
}

/* ═══════════════ Eval 面板 ═══════════════ */

async function loadRuns() {
  const sel = $("#eval-select");
  try {
    const runs = await (await fetch("/api/eval")).json();
    sel.innerHTML = runs.map((r) =>
      `<option value="${esc(r.filename)}">${esc(r.timestamp)} · ${esc(r.config.llm_model || "?")} · full 4avg ${pct(r.summary.full_4avg, 2)}</option>`
    ).join("");
    if (runs.length) { sel.selectedIndex = 0; loadRun(sel.value); }
    else $("#eval-summary").innerHTML = '<div class="muted">暂无 eval 结果</div>';
  } catch (e) {
    $("#eval-summary").innerHTML = `<div class="muted">加载失败: ${esc(String(e))}</div>`;
  }
}

$("#eval-select").onchange = (e) => loadRun(e.target.value);
$("#btn-refresh").onclick = loadRuns;

async function loadRun(filename) {
  const d = await (await fetch("/api/eval/" + encodeURIComponent(filename))).json();
  renderSummary(d.summary || {});
  renderTable(d);
}

function renderSummary(s) {
  const cards = [
    ["Baseline 4avg", pct(s.baseline_4avg, 2), s.baseline_4avg >= 0.75 ? "good" : "bad"],
    ["Draft 4avg", pct(s.draft_4avg, 2), s.draft_4avg >= 0.75 ? "good" : "bad"],
    ["Full 4avg", pct(s.full_4avg, 2), s.full_4avg >= 0.75 ? "good" : "bad"],
    ["Δ vs Baseline", (s.avg_delta_vs_baseline != null ? "+" : "") + (s.avg_delta_vs_baseline ?? 0).toFixed(4), s.avg_delta_vs_baseline > 0 ? "good" : "bad"],
    ["Δ Draft→Full", "+" + (s.avg_delta_draft_vs_full ?? 0).toFixed(4), s.avg_delta_draft_vs_full > 0 ? "good" : "bad"],
    ["accept / revise", `${s.verdict_distribution?.accept ?? 0} / ${s.verdict_distribution?.revise ?? 0}`, "neutral"],
    ["平均修订轮次", s.avg_correction_rounds?.toFixed(2) ?? "—", "neutral"],
    ["改进率 / 退化率", `${pct(s.improvement_rate, 0)} / ${pct(s.degradation_rate, 0)}`, s.improvement_rate > s.degradation_rate ? "good" : "bad"],
  ];
  $("#eval-summary").innerHTML = cards.map(([k, v, cls]) =>
    `<div class="s-card"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`).join("");
}

function renderTable(d) {
  const cols = ["base", "draft", "full"].map((g) => {
    const m = d.per_question[0] && d.per_question[0][g];
    const keys = m ? ["faithfulness", "relevancy", "precision", "recall"] : [];
    return keys;
  });

  let html = `<table class="eval-table"><thead><tr>
    <th rowspan="2">题</th>
    <th colspan="4">Baseline</th><th colspan="4">Draft</th><th colspan="4">Full</th>
    <th rowspan="2">裁决</th><th rowspan="2">修订</th><th rowspan="2">Δ Full</th>
  </tr><tr>${["baseline", "draft", "full"].map((g) => cols).length
    ? '<th class="col-group">F</th><th class="col-group">R</th><th class="col-group">P</th><th class="col-group">C</th>'.repeat(3)
    : ""}
  </tr></thead><tbody>`;

  d.per_question.forEach((q) => {
    const [b, dr, f] = ["baseline", "draft", "full"].map((g) => q[g]);
    const avg = (x) => x ? ((x.faithfulness + x.relevancy + x.precision + x.recall) / 4).toFixed(3) : "—";
    const delta = b && f ? f.avg - b.avg : null;
    const deltaCls = delta > 0.01 ? "diff-up" : delta < -0.01 ? "diff-down" : "";
    html += `<tr data-q="${esc(q.id)}">
      <td><b>${esc(q.id)}</b></td>
      ${[b, dr, f].map((g) => g
        ? `<td class="num">${(g.faithfulness ?? 0).toFixed(2)}</td><td class="num">${(g.relevancy ?? 0).toFixed(2)}</td><td class="num">${(g.precision ?? 0).toFixed(2)}</td><td class="num">${(g.recall ?? 0).toFixed(2)}</td>`
        : '<td colspan="4">—</td>').join("")}
      <td>${esc(q.difficulty || "")}</td>
      <td>${esc(String(q.debate_stats?.correction_rounds ?? 0))}</td>
      <td class="num ${deltaCls}">${delta != null ? (delta > 0 ? "+" : "") + delta.toFixed(3) : "—"}</td>
    </tr>`;
  });
  html += "</tbody></table>";
  $("#eval-table-wrap").innerHTML = html;

  // 行点击 → 显示该题答案对比
  $("#eval-table-wrap").querySelectorAll("tr[data-q]").forEach((tr) => {
    tr.onclick = () => showQuestionDetail(d, tr.dataset.q);
  });
}

function showQuestionDetail(d, qid) {
  const q = d.per_question.find((x) => x.id === qid);
  if (!q) return;
  const mk = (g, label) => `
    <div class="s-card" style="grid-column:auto">
      <div class="k">${label}</div>
      <div class="answer-text" style="max-height:260px;overflow-y:auto">${esc(q[g].answer || "")}</div>
    </div>`;
  $("#eval-detail").innerHTML = `
    <h3 style="font-size:14px;margin:14px 0 8px">${esc(q.id)} · ${esc(q.question)}</h3>
    <div class="summary-cards">${mk("baseline", "Baseline 答案")}${mk("draft", "Draft 草稿")}${mk("full", "Full 最终")}</div>`;
  $("#eval-detail").scrollIntoView({ block: "nearest" });
}

/* ═══════════════ 启动 ═══════════════ */
checkHealth();
