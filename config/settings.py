"""全局配置 — DeepReason 多Agent自纠错推理引擎

所有可配置的常量集中管理，方便切换模型/数据库/环境。
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# LLM 配置 (DeepSeek API，兼容 OpenAI SDK)
# ============================================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# 各Agent使用的模型名称（API 仅支持 deepseek-v4-pro / deepseek-v4-flash）
# 与当前 3-Agent 架构对应：Planner / Generator / Critic / Reviser
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "deepseek-v4-flash")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "deepseek-v4-pro")   # Generator：答案生成（原 ADVOCATE_MODEL）
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "deepseek-v4-flash")       # Critic：审查+裁决+工具核查（原 JUDGE_MODEL）
REVISER_MODEL = os.getenv("REVISER_MODEL", "deepseek-v4-flash")     # Reviser：按审稿意见修订（原 SKEPTIC_MODEL）


# ============================================================================
# Milvus 向量库配置 (Docker Standalone)
# ============================================================================

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "deepreason_knowledge")

# 稠密向量维度 (bge-small-zh-v1.5: 512维; BGE-M3: 1024维)
DENSE_DIM = 1024


# ============================================================================
# Embedding 模型配置
# ============================================================================

# 大模型 (BGE-M3, 1024维)，多语言 + 稠密/稀疏双表征
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-m3",
)

# 批量处理时的batch size（根据显存/内存调整）
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))


# ============================================================================
# 重排序模型配置
# ============================================================================

# Cross-Encoder 重排序模型
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "BAAI/bge-reranker-v2-m3",
)

# 重排序时保留的Top-N候选数
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "10"))


# ============================================================================
# Parent-Child 双层 Chunk 配置
# ============================================================================

# Parent Chunk（大块，喂给 LLM 的上下文）— 语义边界切分，信息完整
PARENT_MAX_CHARS = int(os.getenv("PARENT_MAX_CHARS", "3000"))
PARENT_MIN_CHARS = int(os.getenv("PARENT_MIN_CHARS", "500"))

# Child Chunk（小块，做 embedding 索引）— 精准检索单元
# 目标：每个 Parent 约拆分为 3 个 Child（2000/600 ≈ 3.3）
CHILD_TARGET_CHARS = int(os.getenv("CHILD_TARGET_CHARS", "600"))
CHILD_MIN_CHARS = int(os.getenv("CHILD_MIN_CHARS", "200"))

# Child→Parent 扩展阈值：Top-K 命中结果中，当 ≥N 个 child
# 来自同一个 parent 时，返回该 parent 替代这些 child
PARENT_EXPANSION_THRESHOLD = int(os.getenv("PARENT_EXPANSION_THRESHOLD", "2"))


# ============================================================================
# HyDE 查询扩展配置
# ============================================================================

# 是否启用 HyDE（查询时生成假设性答案辅助检索）
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "true").lower() == "true"

# HyDE 使用的模型（轻量任务用 flash 即可；API 仅支持 deepseek-v4-pro / deepseek-v4-flash）
HYDE_MODEL = os.getenv("HYDE_MODEL", "deepseek-v4-flash")

# HyDE 生成温度（假设性答案不需要精确，稍高温度增加多样性）
HYDE_TEMPERATURE = float(os.getenv("HYDE_TEMPERATURE", "0.4"))

# Multi-HyDE 不同视角数量（Advocate/Skeptic 各自生成不同假设）
MULTI_HYDE_PERSPECTIVES = ["neutral", "supportive", "critical"]


# ============================================================================
# 混合检索参数
# ============================================================================

# 稠密向量检索召回数（在 Child Chunk 层级检索）
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "20"))

# BM25稀疏检索召回数（在 Parent Chunk 层级检索）
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "15"))

# 融合后返回给后续流程的最终文档数
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

# 稠密/稀疏加权融合的权重（dense_weight + sparse_weight = 1.0）
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.7"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.3"))


# ============================================================================
# Guardrails 上限常量
# ============================================================================

MAX_CORRECTION_ROUNDS = 3      # Reflexion 自纠错最大轮数
MAX_DEBATE_ROUNDS = 2          # 辩论最大轮数
MAX_RETRIEVAL_HOPS = 3         # 多跳检索最大跳数

# 单次 LLM 调用超时（秒）——网络/网关卡死时兜底，避免整个会话挂起
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))

# 收敛检测：修订前后答案相似度 ≥ 该阈值视为"未实质变化"，提前终止循环
CONVERGENCE_SIMILARITY_THRESHOLD = float(os.getenv("CONVERGENCE_SIMILARITY_THRESHOLD", "0.85"))

# 低置信度标注阈值：最终答案 confidence 低于此值时附加不确定性提示
LOW_CONFIDENCE_ANNOTATION_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_ANNOTATION_THRESHOLD", "0.6"))


# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 处理后数据路径
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHUNKS_FILE = os.path.join(PROCESSED_DIR, "all_chunks.json")
PARENT_CHUNKS_FILE = os.path.join(PROCESSED_DIR, "parent_chunks.json")
CHILD_CHUNKS_FILE = os.path.join(PROCESSED_DIR, "child_chunks.json")

# BM25 索引持久化路径（在 Parent Chunk 层级构建）
BM25_INDEX_DIR = os.path.join(PROCESSED_DIR, "bm25_index")

# 评估数据路径
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")


# ============================================================================
# Agent 工具（MCP）配置
# ============================================================================

# Critic 工具核查：对低置信度的 unsupported/contradicted 断言，先用工具
# 在全语料查证再决定是否保留该 issue（拦截"证据只是没被检索到"的假阳性）
CRITIC_TOOL_VERIFY_ENABLED = os.getenv("CRITIC_TOOL_VERIFY_ENABLED", "true").lower() == "true"

# 只核查置信度低于此值的存疑断言（≥0.9 的字面确证级不再核查）
CRITIC_TOOL_VERIFY_CONF_THRESHOLD = float(os.getenv("CRITIC_TOOL_VERIFY_CONF_THRESHOLD", "0.9"))

# 每轮审查最多工具核查的断言数（LLM 调用成本封顶）
CRITIC_TOOL_VERIFY_MAX_CLAIMS = int(os.getenv("CRITIC_TOOL_VERIFY_MAX_CLAIMS", "2"))
