"""Global configuration for DeepReason project."""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# Default model names
PLANNER_MODEL = "deepseek-v4-pro"          # 任务分解，用性价比高的模型
ADVOCATE_MODEL = "deepseek-v4-pro"         # 辩论论证，并发量大用便宜模型
SKEPTIC_MODEL = "deepseek-v4-pro"          # 反驳论证
JUDGE_MODEL = "deepseek-v4-pro"            # 裁决（如果预算允许可换成 claude-sonnet-4-6）
VALIDATOR_MODEL = "deepseek-v4-pro"        # Reflexion校验

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION_NAME = "deepreason_knowledge"

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
EMBEDDING_DIM = 1024  # BGE-M3 default dimension

# Guardrails
MAX_CORRECTION_ROUNDS = int(os.getenv("MAX_CORRECTION_ROUNDS", "3"))
MAX_DEBATE_ROUNDS = int(os.getenv("MAX_DEBATE_ROUNDS", "2"))
MAX_RETRIEVAL_HOPS = int(os.getenv("MAX_RETRIEVAL_HOPS", "3"))

# Retrieval
HYBRID_SEARCH_TOP_K = 10          # 混合检索返回数量
RERANK_TOP_K = 5                  # 重排序后保留数量
BM25_THRESHOLD = 0.85             # BM25归一化阈值

# Evaluation
EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "data/eval/questions.json")