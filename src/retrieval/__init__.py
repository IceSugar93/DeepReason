"""检索模块 — Parent-Child + HyDE + BM25 + Cross-Encoder 重排序

Advanced RAG 检索架构:
1. HyDE: LLM 生成假设性答案，桥接查询-文档语义鸿沟
2. Child Chunk Dense 检索: 在 Milvus 中精准匹配小片段
3. Child→Parent 扩展: 按 parent_id 聚合，返回完整上下文
4. BM25 关键词检索: 在 Parent 层级做稀疏匹配
5. RRF 融合: 合并稠密+稀疏两路结果
6. Cross-Encoder 重排序: 精细相关性打分
"""

from src.retrieval.vector_store import (
    connect_milvus,
    disconnect_milvus,
    create_collection,
    get_collection,
    insert_embeddings,
    search_dense,
    expand_children_to_parents,
    load_parents,
    lookup_parents,
    get_collection_stats,
)
from src.retrieval.bm25_store import BM25Store
from src.retrieval.reranker import Reranker
from src.retrieval.hyde import HyDE
from src.retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    # 向量存储
    "connect_milvus",
    "disconnect_milvus",
    "create_collection",
    "get_collection",
    "insert_embeddings",
    "search_dense",
    "expand_children_to_parents",
    "load_parents",
    "lookup_parents",
    "get_collection_stats",
    # 检索器
    "BM25Store",
    "Reranker",
    "HyDE",
    "HybridRetriever",
]
