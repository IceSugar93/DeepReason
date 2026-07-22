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

# 各Agent使用的模型名称（DeepSeek 目前主要为 deepseek-chat）
# 后续可替换为不同模型以优化成本/效果
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "deepseek-v4-pro")
ADVOCATE_MODEL = os.getenv("ADVOCATE_MODEL", "deepseek-v4-pro")
SKEPTIC_MODEL = os.getenv("SKEPTIC_MODEL", "deepseek-v4-pro")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-v4-pro")
VALIDATOR_MODEL = os.getenv("VALIDATOR_MODEL", "deepseek-v4-pro")


# ============================================================================
# Milvus 向量库配置 (Docker Standalone)
# ============================================================================

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "deepreason_knowledge")

# 稠密向量维度 (bge-small-zh-v1.5: 512维; BGE-M3: 1024维)
DENSE_DIM = 512


# ============================================================================
# Embedding 模型配置
# ============================================================================

# 轻量模型快速跑通全流程，验证完换回 BAAI/bge-m3 (1024维)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-small-zh-v1.5",
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

# HyDE 使用的模型（与主 LLM 一致即可，消耗很小）
HYDE_MODEL = os.getenv("HYDE_MODEL", "deepseek-chat")

# HyDE 生成温度（假设性答案不需要精确，稍高温度增加多样性）
HYDE_TEMPERATURE = float(os.getenv("HYDE_TEMPERATURE", "0.4"))

# Multi-HyDE 不同视角数量（Advocate/Skeptic 各自生成不同假设）
MULTI_HYDE_PERSPECTIVES = ["neutral", "supportive", "critical"]


# ============================================================================
# 混合检索参数
# ============================================================================

# 稠密向量检索召回数（在 Child Chunk 层级检索）
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "30"))

# BM25稀疏检索召回数（在 Parent Chunk 层级检索）
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "20"))

# 融合后返回给后续流程的最终文档数
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "8"))

# 稠密/稀疏加权融合的权重（dense_weight + sparse_weight = 1.0）
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.4"))


# ============================================================================
# Guardrails 上限常量
# ============================================================================

MAX_CORRECTION_ROUNDS = 3      # Reflexion 自纠错最大轮数
MAX_DEBATE_ROUNDS = 2          # 辩论最大轮数
MAX_RETRIEVAL_HOPS = 3         # 多跳检索最大跳数


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
