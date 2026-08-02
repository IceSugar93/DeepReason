"""混合检索器 — HyDE + Parent-Child + BM25 + Cross-Encoder 重排序

完整检索流程（高级 RAG）:
1. HyDE: LLM 生成假设性答案，桥接查询-文档语义鸿沟
2. Dense: 用假设答案的 embedding 在 Child Chunks 上做精准检索
3. Expand: Child 结果按 parent_id 聚合 → 映射为 Parent Chunks
4. BM25: 在 Parent Chunks 上做关键词检索（独立通道）
5. RRF: 融合 Dense（映射后）+ BM25 结果
6. Rerank: Cross-Encoder 精细重排序
7. 返回 Top-K Parent Chunks
"""

import numpy as np

from config.settings import (
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RERANKER_TOP_K,
    FINAL_TOP_K,
    PARENT_EXPANSION_THRESHOLD,
    HYDE_ENABLED,
    MULTI_HYDE_PERSPECTIVES,
)
from src.retrieval.vector_store import (
    search_dense,
    expand_children_to_parents,
)
from src.retrieval.bm25_store import BM25Store
from src.retrieval.reranker import Reranker
from src.retrieval.hyde import HyDE


# ============================================================================
# RRF 融合（在 Parent 层级）
# ============================================================================

def _rrf_fusion_parent_level(
    dense_parents: list[dict],
    sparse_parents: list[dict],
    k: int = 60,
) -> list[dict]:
    """RRF 融合 — 在 Parent 层级合并稠密和稀疏检索结果。

    稠密通道: Child 检索 → 映射为 Parent（已有 child_count 字段）
    稀疏通道: BM25 直接在 Parent 上检索

    两个通道的结果都已经是 Parent 层级，直接按 parent_id 融合。

    Args:
        dense_parents: 稠密通道的 parent 结果（已做 expand_children_to_parents）。
        sparse_parents: BM25 通道的 parent 结果。
        k: RRF 参数，默认 60。

    Returns:
        融合后的 parent 文档列表。
    """
    merged: dict[str, dict] = {}

    for rank, doc in enumerate(dense_parents):
        cid = doc["chunk_id"]
        merged[cid] = {**doc, "dense_score": doc["score"], "bm25_score": 0.0}
        merged[cid]["rrf_score"] = 1.0 / (k + rank + 1)

    for rank, doc in enumerate(sparse_parents):
        cid = doc["chunk_id"]
        rrf_contrib = 1.0 / (k + rank + 1)
        if cid in merged:
            merged[cid]["bm25_score"] = doc["score"]
            merged[cid]["rrf_score"] += rrf_contrib
            # 优先使用 parent chunk 中的完整 content
            if len(doc.get("content", "")) > len(merged[cid].get("content", "")):
                merged[cid]["content"] = doc["content"]
        else:
            merged[cid] = {**doc, "dense_score": 0.0, "bm25_score": doc["score"]}
            merged[cid]["rrf_score"] = rrf_contrib

    # 用 RRF 分作为 score
    for doc in merged.values():
        doc["score"] = doc["rrf_score"]

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)


# ============================================================================
# 混合检索器（升级版）
# ============================================================================

class HybridRetriever:
    """混合检索器 — HyDE + Parent-Child + BM25 + 重排序。

    对外暴露单一 search() 接口，内部编排 6 步高级 RAG 流程。
    设计为有状态对象——持有 BM25 索引、Reranker 模型和 HyDE 扩展器。

    使用示例:
        retriever = HybridRetriever(bm25_store)
        results = retriever.search("什么是 MCP 协议？", query_emb, top_k=8)
    """

    def __init__(
        self,
        bm25_store: BM25Store,
        reranker: Reranker | None = None,
        hyde: HyDE | None = None,
        fusion_method: str = "rrf",
        enable_hyde: bool = HYDE_ENABLED,
    ):
        """初始化混合检索器。

        Args:
            bm25_store: 已构建的 BM25 检索器（在 Parent Chunks 层级）。
            reranker: Cross-Encoder 重排序器。
            hyde: HyDE 查询扩展器。
            fusion_method: 融合策略，当前仅支持 "rrf"。
            enable_hyde: 是否启用 HyDE 查询扩展（默认启用）。
        """
        self.bm25 = bm25_store
        self.reranker = reranker or Reranker()
        self.hyde = hyde or HyDE()
        self.fusion_method = fusion_method
        self.enable_hyde = enable_hyde

    # ------------------------------------------------------------------
    # 检索入口
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = FINAL_TOP_K,
        doc_type_filter: str | None = None,
        enable_rerank: bool = True,
        enable_multi_hyde: bool = False,
        skip_hyde: bool = False,
    ) -> list[dict]:
        """执行完整的 6 步高级 RAG 检索。

        Args:
            query: 用户原始查询。
            query_embedding: 原始查询的稠密向量（不使用 HyDE 时的备选）。
            top_k: 最终返回的 Parent Chunk 数。
            doc_type_filter: 可选，按文档类型过滤。
            enable_rerank: 是否启用 Cross-Encoder 重排序。
            enable_multi_hyde: 是否启用 Multi-HyDE（多视角多路检索融合）。
            skip_hyde: 跳过 HyDE 查询扩展（工具调用场景：查询已是具体文本，
                省去每次一次 LLM 假设答案生成的开销）。

        Returns:
            Parent Chunk 列表，每个带完整元数据和各级分数。
        """
        # ================================================================
        # Step 1: HyDE 查询扩展
        # ================================================================
        if self.enable_hyde and not skip_hyde and enable_multi_hyde:
            # Multi-HyDE: 多视角独立检索 → RRF 融合
            return self._search_multi_hyde(
                query, query_embedding, top_k, doc_type_filter, enable_rerank
            )

        if self.enable_hyde and not skip_hyde:
            # 单视角 HyDE: 生成假设答案 → embed → 检索
            hyde_query = self.hyde.expand_query(query)
            hyde_embedding = self._embed_query(hyde_query)
            dense_query_emb = hyde_embedding
            used_hyde = True
        else:
            dense_query_emb = query_embedding
            used_hyde = False

        # ================================================================
        # Step 2: 稠密向量检索（在 Child Chunks 层级）
        # ================================================================
        child_results = search_dense(
            query_embedding=dense_query_emb,
            top_k=DENSE_TOP_K,
            doc_type_filter=doc_type_filter,
        )

        # ================================================================
        # Step 3: Child → Parent 扩展
        # ================================================================
        dense_parents = expand_children_to_parents(
            child_results,
            threshold=PARENT_EXPANSION_THRESHOLD,
        )

        # ================================================================
        # Step 4: BM25 检索（在 Parent Chunks 层级）
        # ================================================================
        # BM25 用原始查询（不经过 HyDE），关键词匹配更直接
        sparse_parents = self.bm25.search(query=query, top_k=SPARSE_TOP_K)

        # ================================================================
        # Step 5: RRF 融合
        # ================================================================
        merged = _rrf_fusion_parent_level(dense_parents, sparse_parents)

        # ================================================================
        # Step 6: Cross-Encoder 重排序
        # ================================================================
        if enable_rerank and merged:
            candidates = merged[:RERANKER_TOP_K]
            merged = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            merged = merged[:top_k]

        # 标记是否使用了 HyDE
        for doc in merged:
            doc["hyde_used"] = used_hyde

        return merged

    # ------------------------------------------------------------------
    # Multi-HyDE: 多视角多路检索
    # ------------------------------------------------------------------

    def _search_multi_hyde(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int,
        doc_type_filter: str | None,
        enable_rerank: bool,
    ) -> list[dict]:
        """Multi-HyDE 检索：从 neutral/supportive/critical 三个视角各自检索。

        每条视角独立走「HyDE → 检索 Child → 映射 Parent → BM25 → 融合」，
        最后将三条通道的 Parent 结果做跨视角 RRF 融合。
        """
        perspectives = MULTI_HYDE_PERSPECTIVES
        all_channel_results: list[list[dict]] = []

        for perspective in perspectives:
            # 生成该视角的假设答案 + 扩展查询
            hyde_query = self.hyde.expand_query(query)
            # 为不同视角微调（通过 HyDE 内部重新生成）
            hypothesis = self.hyde.generate(query, perspective=perspective)
            hyde_query_full = f"{hyde_query}\n\n{hypothesis}"

            hyde_emb = self._embed_query(hyde_query_full)

            # 检索 Child → 映射 Parent
            child_results = search_dense(
                query_embedding=hyde_emb,
                top_k=DENSE_TOP_K,
                doc_type_filter=doc_type_filter,
            )
            dense_parents = expand_children_to_parents(
                child_results,
                threshold=PARENT_EXPANSION_THRESHOLD,
            )

            # BM25（始终用原始查询）
            sparse_parents = self.bm25.search(query=query, top_k=SPARSE_TOP_K)

            # 融合
            channel_merged = _rrf_fusion_parent_level(dense_parents, sparse_parents)
            all_channel_results.append(channel_merged)

        # 跨视角 RRF 融合
        cross_channel_merged: dict[str, dict] = {}
        k = 60
        for channel_idx, channel_results in enumerate(all_channel_results):
            for rank, doc in enumerate(channel_results):
                cid = doc["chunk_id"]
                rrf_contrib = 1.0 / (k + rank + 1)
                if cid in cross_channel_merged:
                    cross_channel_merged[cid]["score"] += rrf_contrib
                    cross_channel_merged[cid]["hyde_channel_count"] += 1
                else:
                    cross_channel_merged[cid] = {**doc}
                    cross_channel_merged[cid]["score"] = rrf_contrib
                    cross_channel_merged[cid]["hyde_channel_count"] = 1

        merged = sorted(
            cross_channel_merged.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        # 重排序
        if enable_rerank and merged:
            candidates = merged[:RERANKER_TOP_K]
            merged = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            merged = merged[:top_k]

        for doc in merged:
            doc["hyde_used"] = True
            doc["multi_hyde"] = True

        return merged

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def _embed_query(self, text: str) -> np.ndarray:
        """将文本转为 BGE-M3 稠密向量（复用检索时的 embedding 模型）。"""
        # 这里需要 SentenceTransformer 实例。
        # 为避免重复加载，通过延迟导入 + 模块级缓存。
        return _get_embedding(text)


    def search_multi_hop(
        self,
        query: str,
        query_embedding: np.ndarray,
        previous_context: str = "",
        top_k: int = FINAL_TOP_K,
    ) -> list[dict]:
        """多跳检索 — 结合前一步上下文进行检索。

        Args:
            query: 当前跳的查询文本。
            query_embedding: 查询的稠密向量。
            previous_context: 前几跳累积的上下文。
            top_k: 返回文档数。
        """
        if previous_context:
            enhanced_query = f"{query}\n\n参考上下文:\n{previous_context[:500]}"
        else:
            enhanced_query = query

        return self.search(
            query=enhanced_query,
            query_embedding=query_embedding,
            top_k=top_k,
        )


# ============================================================================
# 模块级 embedding 缓存
# ============================================================================

_embed_model = None

def _get_embedding(text: str) -> np.ndarray:
    """获取单条文本的 BGE-M3 稠密向量（模块级模型缓存）。"""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        from config.settings import EMBEDDING_MODEL_NAME
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embed_model.encode(
        text,
        normalize_embeddings=True,
    ).astype(np.float32)
