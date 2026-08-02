# DeepReason — 多Agent自纠错推理引擎

基于 LangGraph 状态机编排的 **Agentic RAG 推理引擎**：检索链（Planner）+ 3-Agent 审查-修订闭环（Generator / Critic / Reviser），配备 MCP 风格工具核查与 Guardrails（收敛检测、安全标注、超时、reject 回退）。

```
retrieve → plan ──→ (multi_hop_retrieve) ──→ generator
    → critic ──accept──→ finalize
       │
       └──revise──→ reviser ──→ verify ──→ (critic 重审 / reviser 继续 / finalize)
```

## 快速开始

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. 创建 .env（API Key / Milvus / 模型，见下方）

# 3. 启动 Milvus（Docker Standalone）
docker compose -f scripts/milvus-standalone-docker-compose.yml up -d

# 4. 构建知识库索引（arXiv 论文 + 框架文档 + 博客）
python scripts/collect_papers.py        # 可选：采集数据
python scripts/build_index.py           # 切块 + Embedding + 建索引

# 5. 冒烟测试
python scripts/test_retrieval.py        # 检索链路
python scripts/test_graph.py            # 推理图端到端
```

`.env` 关键配置：

```ini
DEEPSEEK_API_KEY=sk-xxx
# 各 Agent 模型（API 仅支持 deepseek-v4-pro / deepseek-v4-flash）
GENERATOR_MODEL=deepseek-v4-flash
CRITIC_MODEL=deepseek-v4-flash
REVISER_MODEL=deepseek-v4-flash
PLANNER_MODEL=deepseek-v4-flash
# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

## 评估

```bash
# 三组对照评估（Baseline / Draft / Full），结果写入 eval/results/
python eval/run_eval_debate.py --limit 3     # 冒烟（前 3 题）
python eval/run_eval_debate.py               # 全量 10 题
```

## 评估结果

**评测口径**（2026-07 修正后固定）：
- LLM-as-Judge 逐条分类判定，score 由程序按比率计算（非自由打分，可复现）
- Baseline 的 faithfulness 对照「外层单跳检索集」评估；Draft/Full 对照「引擎内部实际使用的检索文档集」（含多跳合并）评估
- 指标：Faithfulness / Answer Relevancy / Context Precision / Context Recall（4avg 为均值）

**迭代演进**（10 题全量，`eval/results/` 可复现）：

| 日期 | 生成模型 | Baseline | Draft | Full | 备注 |
|---|---|---|---|---|---|
| 07-23 | pro | 0.671 | 0.638 | 0.542 | 早期：裁决链路不稳定，3/10 非 accept |
| 07-24 | pro | 0.784 | 0.779 | 0.791 | 链路稳定，Full 整体 ≥ Draft |
| 07-25 | pro | 0.723 | 0.759 | 0.757 | 评测口径修正后，Draft 稳定超越 Baseline |
| 07-27 | pro | 0.768 | 0.761 | 0.762 | 全量首轮通过 |
| 08-01 | flash | 0.781 | **0.794** | 0.794 | 最佳成绩：Draft 全维度超越 Baseline |
| 08-02 | pro | 0.761 | 0.716 | 0.718 | 审查-修订首次大规模触发（4/10 revise）；修订 3/4 在忠实度上正向改进；收敛检测在相似度 0.93 时提前终止 |

> 说明：08-02 分数回落源于生成模型切换（pro 在该问题集上弱于 flash 正式版）。触发修订的题目均未被改坏——机制有效性（修订正确率、收敛检测、工具核查）以过程数据为准，而非仅看 4avg。

## Web 界面

可视化推理全过程（流式展示各节点执行）与历史 eval 结果：

```bash
uvicorn web.app:app --port 8010
# 打开 http://127.0.0.1:8010
```

- **推理演示**：输入问题 → SSE 实时推送 retrieve / plan / generator / critic（issues、工具核查）/ reviser（收敛检测）/ verify 各节点 → 最终答案
- **Eval 结果**：历史运行列表、逐题四指标（Faithfulness / Relevancy / Precision / Recall）、verdict 分布、修订轨迹，点击行展开答案对比

## 技术栈

| 层 | 选型 |
|---|---|
| 编排 | LangGraph（状态机 + 条件路由 + 检查点） |
| LLM | DeepSeek（deepseek-v4-pro / flash，OpenAI 兼容 SDK） |
| 检索 | Milvus + BGE-M3 稠密检索 + BM25 稀疏检索 + bge-reranker 重排 + HyDE/Multi-HyDE 查询扩展 |
| 结构 | Parent-Child 双层 Chunk |
| 工具 | MCP 风格 4 工具（paper_search / concept_query / framework_doc_search / compare），Critic 断言核查 |
| 评测 | LLM-as-Judge 逐条判定（faithfulness / relevancy / precision / recall） |
| Web | FastAPI + SSE + 原生单页前端 |
