"""BM25 稀疏检索 — 基于 jieba 分词 + 倒排索引

实现经典的 BM25 算法作为稠密向量的互补检索通道。
在 Parent-Child 架构中，BM25 在 Parent Chunk 层级构建索引——
关键词匹配需要完整上下文才能有效，所以用 Parent 而非 Child。
"""
import json
import math
import os
import pickle
from collections import defaultdict

import jieba

from config.settings import BM25_INDEX_DIR


# ============================================================================
# BM25 算法核心
# ============================================================================

class BM25Store:
    """BM25 稀疏检索器。

    使用 jieba 进行中文分词，构建倒排索引。
    支持索引持久化，避免每次重新构建。

    BM25 公式:
        score(D, Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))

    其中:
        - IDF(qi): 逆文档频率
        - f(qi,D): 词 qi 在文档 D 中的词频
        - |D|: 文档长度
        - avgdl: 平均文档长度
        - k1, b: 调节参数（默认 k1=1.5, b=0.75）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """初始化 BM25 检索器。

        Args:
            k1: 词频饱和度调节参数，默认 1.5。
            b: 文档长度归一化参数，默认 0.75。
        """
        self.k1 = k1
        self.b = b

        # 索引数据结构
        self.chunks: list[dict] = []                          # 所有文档
        self.doc_lengths: list[int] = []                      # 每条文档的 token 数
        self.avgdl: float = 0.0                               # 平均文档长度
        self.doc_freq: dict[str, int] = defaultdict(int)      # 词 → 出现该词的文档数
        self.inverted_index: dict[str, dict[int, int]] = defaultdict(dict)  # 词 → {doc_id: 词频}
        self.idf: dict[str, float] = {}                       # 词 → IDF值
        self._built = False

    # ------------------------------------------------------------------
    # 分词
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """对文本进行分词处理。

        中文使用 jieba 精确模式，英文/数字保留原样。
        过滤掉单字和纯空白 token。
        """
        tokens = jieba.lcut(text.lower())
        # 过滤：去除空白字符和长度不足的 token
        return [t.strip() for t in tokens if len(t.strip()) >= 2]

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[dict]) -> None:
        """从 chunk 列表构建 BM25 倒排索引。

        Args:
            chunks: chunk 列表，每个元素需要 content 字段。
        """
        self.chunks = chunks
        total_tokens = 0

        for doc_id, chunk in enumerate(chunks):
            tokens = self.tokenize(chunk["content"])
            self.doc_lengths.append(len(tokens))
            total_tokens += len(tokens)

            # 计算词频分布
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            # 写入倒排索引
            for token, freq in tf.items():
                self.inverted_index[token][doc_id] = freq
                self.doc_freq[token] += 1  # 每个文档最多贡献 1 次

        # 计算平均文档长度
        num_docs = len(chunks)
        self.avgdl = total_tokens / num_docs if num_docs > 0 else 1.0

        # 预计算 IDF
        self._compute_idf(num_docs)

        self._built = True

    def _compute_idf(self, num_docs: int) -> None:
        """预计算所有词的 IDF 值。

        IDF(q) = log((N - df(q) + 0.5) / (df(q) + 0.5) + 1)

        其中 N 是文档总数，df(q) 是出现词 q 的文档数。
        """
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log(
                (num_docs - df + 0.5) / (df + 0.5) + 1
            )

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 检索。

        Args:
            query: 查询文本。
            top_k: 返回 Top-K 结果。

        Returns:
            [{chunk_id, content, source, doc_type, title, score}, ...]
        """
        if not self._built:
            raise RuntimeError("索引尚未构建，请先调用 build_index()")

        query_tokens = self.tokenize(query)

        # 计算每个候选文档的 BM25 分
        scores: dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf.get(token, 0.0)
            for doc_id, tf in self.inverted_index[token].items():
                doc_len = self.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )
                scores[doc_id] += idf * numerator / denominator

        # 按分数降序排序，取 Top-K
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, bm25_score in ranked:
            chunk = self.chunks[doc_id]
            results.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "content": chunk.get("content", ""),
                "source": chunk.get("source", ""),
                "doc_type": chunk.get("doc_type", ""),
                "title": chunk.get("title", ""),
                "score": bm25_score,
            })

        return results

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, save_dir: str = BM25_INDEX_DIR) -> None:
        """将 BM25 索引保存到磁盘。

        保存三个文件:
            - chunks.pkl: 文档列表
            - index.pkl: 倒排索引、文档频率、IDF 等
            - params.json: 配置参数 (k1, b, 统计信息)
        """
        if not self._built:
            raise RuntimeError("索引尚未构建，请先调用 build_index()")

        os.makedirs(save_dir, exist_ok=True)

        # 文档原文（pickle 因为包含嵌套 dict）
        with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        # 索引数据
        index_data = {
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "doc_freq": dict(self.doc_freq),
            "inverted_index": dict(self.inverted_index),
            "idf": self.idf,
        }
        with open(os.path.join(save_dir, "index.pkl"), "wb") as f:
            pickle.dump(index_data, f)

        # 参数（JSON 方便人类阅读）
        params = {
            "k1": self.k1,
            "b": self.b,
            "num_docs": len(self.chunks),
            "avgdl": self.avgdl,
            "vocab_size": len(self.idf),
        }
        with open(os.path.join(save_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

    def load(self, save_dir: str = BM25_INDEX_DIR) -> None:
        """从磁盘加载 BM25 索引。

        Args:
            save_dir: 索引目录路径。

        Raises:
            FileNotFoundError: 索引目录不存在或文件缺失。
        """
        chunks_path = os.path.join(save_dir, "chunks.pkl")
        index_path = os.path.join(save_dir, "index.pkl")
        params_path = os.path.join(save_dir, "params.json")

        for path in [chunks_path, index_path, params_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"索引文件不存在: {path}")

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        with open(index_path, "rb") as f:
            index_data = pickle.load(f)

        self.doc_lengths = index_data["doc_lengths"]
        self.avgdl = index_data["avgdl"]
        self.doc_freq = index_data["doc_freq"]
        self.inverted_index = index_data["inverted_index"]
        self.idf = index_data["idf"]

        self._built = True

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def num_docs(self) -> int:
        return len(self.chunks)
