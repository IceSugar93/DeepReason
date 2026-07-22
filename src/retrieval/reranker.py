"""Cross-Encoder 重排序 — 对检索候选文档做精细相关性打分

混合检索（稠密+稀疏）返回的候选集用 Cross-Encoder 做第二层排序，
相比 embedding 余弦相似度，Cross-Encoder 同时编码 query 和 document，
交互式注意力能更准确地判断语义相关性。
"""

import numpy as np
from sentence_transformers import CrossEncoder

from config.settings import RERANKER_MODEL_NAME


class Reranker:
    """Cross-Encoder 重排序器。

    对混合检索返回的 Top-K 候选文档进行精细相关性打分，
    按分数重新排序后返回。

    使用 BGE-Reranker-v2-M3 模型，该模型对中英文均支持良好。
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        """初始化重排序器。

        Args:
            model_name: Cross-Encoder 模型名称或本地路径。
                       默认使用 BAAI/bge-reranker-v2-m3。
        """
        # 延迟加载，节省内存
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """获取 CrossEncoder 模型实例（延迟加载）。"""
        if self._model is None:
            self._model = CrossEncoder(
                self._model_name,
                max_length=8192,  # BGE-Reranker-v2-M3 支持长文本
                device="cpu",      # 默认 CPU，有 GPU 可改为 "cuda"
            )
        return self._model

    # ------------------------------------------------------------------
    # 重排序
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 8,
    ) -> list[dict]:
        """对候选文档列表做重排序。

        Args:
            query: 查询文本。
            candidates: 候选文档列表，每个元素需要 content 字段。
                       格式: [{chunk_id, content, source, doc_type, title, score}, ...]
            top_k: 返回 Top-K 结果。

        Returns:
            按 Cross-Encoder 相关性分数重新排序后的文档列表。
            每个文档增加 rerank_score 字段，保留原始 score 作为 retrieval_score。
        """
        if not candidates:
            return []

        # 组装 (query, document) 对
        pairs = [(query, doc["content"]) for doc in candidates]

        # Cross-Encoder 打分
        scores: list[float] = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=False,
        )

        # 将分数写入候选文档
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)
            # 保留原始检索分数
            if "score" in doc:
                doc["retrieval_score"] = doc["score"]
            doc["score"] = float(score)  # 用 rerank 分数替换默认 score

        # 按分数降序排序
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]

    # ------------------------------------------------------------------
    # 批量重排序
    # ------------------------------------------------------------------

    def rerank_batch(
        self,
        queries: list[str],
        candidates_batch: list[list[dict]],
        top_k: int = 8,
    ) -> list[list[dict]]:
        """批量重排序（用于评估时加速）。

        Args:
            queries: 查询文本列表。
            candidates_batch: 每个查询对应的候选文档列表。
            top_k: 每个查询返回 Top-K 结果。

        Returns:
            每个查询的重排序结果列表。
        """
        return [
            self.rerank(query, candidates, top_k)
            for query, candidates in zip(queries, candidates_batch)
        ]
