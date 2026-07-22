"""Milvus 向量存储 — Parent-Child 双层级检索

Parent-Child 策略（Milvus 官方推荐的 Small-to-Big 模式）:
- Child Chunk（200-500 chars）存入 Milvus 做 embedding 索引 → 精准检索
- Parent Chunk（1500-3000 chars）存入 JSON 文件 → 检索后按 parent_id 查找，返回完整上下文

检索流程:
  查询 → HyDE 扩展 → embed → 搜 Child Chunks → 按 parent_id 聚合
       → lookup_parents() 获取完整 Parent → 去重 → 返回
"""

import json
import os
from typing import Any

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from config.settings import (
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
    DENSE_DIM,
    PARENT_CHUNKS_FILE,
)


# ============================================================================
# Milvus 连接管理
# ============================================================================

def connect_milvus(host: str = MILVUS_HOST, port: int = MILVUS_PORT,
                   db_name: str = "default") -> None:
    """建立 Milvus 连接（Standalone / Docker 模式）。"""
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            host=host,
            port=port,
            db_name=db_name,
        )


def disconnect_milvus() -> None:
    """断开 Milvus 连接。"""
    if connections.has_connection("default"):
        connections.disconnect("default")


# ============================================================================
# Parent 存储（内存 + JSON）
# ============================================================================

# 全局 parent 缓存: {parent_id: parent_chunk_dict}
_parent_cache: dict[str, dict] = {}


def load_parents(file_path: str = PARENT_CHUNKS_FILE) -> dict[str, dict]:
    """加载 Parent Chunks 到内存。

    Parent chunk 不存入 Milvus（不生成 embedding），而是在检索后
    按 parent_id 查找，用完整内容替换 child 片段。

    Args:
        file_path: parent_chunks.json 路径。

    Returns:
        {parent_id: parent_chunk_dict} 映射表。
    """
    global _parent_cache

    if _parent_cache:
        return _parent_cache

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Parent chunks 文件不存在: {file_path}\n"
            f"请先运行 python src/utils/parsers.py 生成"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        parents = json.load(f)

    _parent_cache = {p["chunk_id"]: p for p in parents}
    return _parent_cache


def lookup_parents(parent_ids: list[str]) -> list[dict]:
    """按 parent_id 列表查找 Parent Chunks。

    Args:
        parent_ids: 需要查找的 parent_id 列表（去重后）。

    Returns:
        对应的 parent chunk dict 列表（保留输入顺序，去重）。
    """
    parents = load_parents()
    seen = set()
    result = []
    for pid in parent_ids:
        if pid not in seen and pid in parents:
            result.append(parents[pid])
            seen.add(pid)
    return result


def get_parent_count() -> int:
    """返回已加载的 parent 总数。"""
    return len(load_parents())


# ============================================================================
# Collection Schema（Child Chunks Only）
# ============================================================================

def get_child_collection_schema() -> CollectionSchema:
    """定义 Child Chunk 的 Milvus Collection Schema。

    字段说明:
    - id: 主键，自动生成 int64
    - chunk_id: 业务主键，如 "paper/1207.0189v1_child0003"
    - content: Child chunk 原文内容
    - dense_vector: BGE-M3 稠密向量（1024维，COSINE 索引）
    - parent_id: 指向 Parent Chunk 的外键（用于 small-to-big 扩展）
    - source: 来源标识
    - doc_type: 文档类型（paper / doc / blog）
    - title: 文档标题
    """
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=256,
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,
        ),
        FieldSchema(
            name="dense_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=DENSE_DIM,
        ),
        # Parent-Child 关联字段
        FieldSchema(
            name="parent_id",
            dtype=DataType.VARCHAR,
            max_length=256,
        ),
        # 元数据字段
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="doc_type",
            dtype=DataType.VARCHAR,
            max_length=32,
        ),
        FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
    ]

    return CollectionSchema(
        fields,
        description="DeepReason 知识库 — Child Chunks（精准检索层）",
    )


# ============================================================================
# Collection 生命周期
# ============================================================================

def create_collection(drop_if_exists: bool = False) -> Collection:
    """创建 Child Chunks 的 Milvus Collection 并构建稠密向量索引。

    Args:
        drop_if_exists: 是否删除已存在的同名 Collection。

    Returns:
        创建好的 Collection 对象。
    """
    connect_milvus()

    if utility.has_collection(MILVUS_COLLECTION):
        if drop_if_exists:
            utility.drop_collection(MILVUS_COLLECTION)
        else:
            col = Collection(MILVUS_COLLECTION)
            col.load()
            return col

    schema = get_child_collection_schema()
    col = Collection(name=MILVUS_COLLECTION, schema=schema)

    # 稠密向量索引（IVF_FLAT + COSINE）
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    col.create_index(field_name="dense_vector", index_params=index_params)

    col.load()
    return col


def get_collection() -> Collection:
    """获取已存在的 Child Collection 实例。"""
    connect_milvus()
    if not utility.has_collection(MILVUS_COLLECTION):
        raise RuntimeError(
            f"Collection '{MILVUS_COLLECTION}' 不存在，请先运行 scripts/build_index.py"
        )
    col = Collection(MILVUS_COLLECTION)
    col.load()
    return col


# ============================================================================
# Child Chunk 插入
# ============================================================================

def insert_embeddings(
    child_chunks: list[dict],
    embeddings: list[np.ndarray],
) -> int:
    """批量插入 Child Chunk 及其稠密向量到 Milvus。

    Args:
        child_chunks: Child chunk 列表，每个元素需要:
                      chunk_id, content, parent_id, source, doc_type, title。
        embeddings: 与 child_chunks 一一对应的稠密向量列表。

    Returns:
        插入的实体总数。
    """
    col = create_collection()
    total = len(child_chunks)
    batch_size = 500

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = child_chunks[start:end]
        batch_embs = embeddings[start:end]

        entities = [
            [c["chunk_id"] for c in batch_chunks],
            [c["content"] for c in batch_chunks],
            [emb.tolist() for emb in batch_embs],
            [c.get("parent_id", c.get("chunk_id", "")) for c in batch_chunks],
            [c.get("source", "") for c in batch_chunks],
            [c.get("doc_type", "") for c in batch_chunks],
            [c.get("title", "") for c in batch_chunks],
        ]

        col.insert(entities)

    col.flush()
    return total


# ============================================================================
# Child Chunk 检索
# ============================================================================

def search_dense(
    query_embedding: np.ndarray,
    top_k: int = 30,
    doc_type_filter: str | None = None,
) -> list[dict]:
    """在 Child Chunks 上做稠密向量相似度检索。

    Args:
        query_embedding: 查询的稠密向量，shape=(DENSE_DIM,)。
        top_k: 返回 Top-K Child 结果。
        doc_type_filter: 可选，按文档类型过滤。

    Returns:
        [{chunk_id, content, parent_id, source, doc_type, title, score}, ...]
    """
    col = get_collection()

    filter_parts = []
    if doc_type_filter:
        filter_parts.append(f'doc_type == "{doc_type_filter}"')
    expr = " and ".join(filter_parts) if filter_parts else None

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

    results = col.search(
        data=[query_embedding.tolist()],
        anns_field="dense_vector",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=[
            "chunk_id", "content", "parent_id",
            "source", "doc_type", "title",
        ],
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "chunk_id": hit.entity.get("chunk_id", ""),
            "content": hit.entity.get("content", ""),
            "parent_id": hit.entity.get("parent_id", ""),
            "source": hit.entity.get("source", ""),
            "doc_type": hit.entity.get("doc_type", ""),
            "title": hit.entity.get("title", ""),
            "score": float(hit.distance),
        })

    return hits


# ============================================================================
# Child → Parent 扩展（核心逻辑）
# ============================================================================

def expand_children_to_parents(
    child_results: list[dict],
    threshold: int = 2,
) -> list[dict]:
    """将检索到的 Child Chunks 映射扩展为 Parent Chunks。

    规则:
    1. 按 parent_id 分组 child 结果。
    2. 同一 parent 下的 child 数 ≥ threshold → 返回该 parent。
    3. 同一 parent 下的 child 数 < threshold → 保留得分最高的 child。
    4. 按每组最高 child 得分排序。

    Args:
        child_results: search_dense() 返回的 child 结果列表。
        threshold: 触发 parent 扩展的阈值（至少 N 个 child 命中才返回 parent）。

    Returns:
        扩展后的结果列表（混合了 parent 和 child）。
        每个元素增加 child_count 字段表示该 parent 下命中 child 的数量。
    """
    # Step 1: 按 parent_id 分组
    groups: dict[str, list[dict]] = {}
    for child in child_results:
        pid = child.get("parent_id", child["chunk_id"])
        if pid not in groups:
            groups[pid] = []
        groups[pid].append(child)

    # Step 2: 对每组决定是返回 parent 还是保留 child
    expanded = []
    for pid, children in groups.items():
        best_child_score = max(c["score"] for c in children)

        if len(children) >= threshold:
            # 多个 child 命中同一个 parent → 返回 parent
            parent = _find_parent(pid)
            if parent:
                expanded.append({
                    "chunk_id": parent["chunk_id"],
                    "content": parent["content"],
                    "source": parent.get("source", children[0].get("source", "")),
                    "doc_type": parent.get("doc_type", children[0].get("doc_type", "")),
                    "title": parent.get("title", children[0].get("title", "")),
                    "parent_id": pid,
                    "score": best_child_score,          # 综合分
                    "child_count": len(children),       # 命中 child 数量
                    "chunk_level": "parent",            # 标记为 parent
                    "matched_children": [c["chunk_id"] for c in children[:5]],
                })
            else:
                # parent 找不到 → 回退，保留最好的 child
                best = max(children, key=lambda c: c["score"])
                best["child_count"] = 1
                best["chunk_level"] = "child"
                expanded.append(best)
        else:
            # 只有 1 个 child 命中 → 保留 child
            best = max(children, key=lambda c: c["score"])
            best["child_count"] = 1
            best["chunk_level"] = "child"
            expanded.append(best)

    # Step 3: 按分数降序排列
    expanded.sort(key=lambda x: x["score"], reverse=True)
    return expanded


def _find_parent(parent_id: str) -> dict | None:
    """在已加载的 parent 缓存中查找。"""
    return load_parents().get(parent_id)


# ============================================================================
# 统计
# ============================================================================

def get_collection_stats() -> dict[str, Any]:
    """获取 Milvus Collection 统计信息。"""
    col = get_collection()
    col.flush()
    num_entities = col.num_entities
    parent_count = get_parent_count()

    stats = {
        "total_children": num_entities,
        "total_parents": parent_count,
        "child_parent_ratio": f"{num_entities / max(parent_count, 1):.1f}:1",
    }

    for doc_type in ["paper", "doc", "blog"]:
        try:
            result = col.query(
                expr=f'doc_type == "{doc_type}"',
                output_fields=["count(*)"],
            )
            stats[f"children_{doc_type}"] = len(result)
        except Exception:
            stats[f"children_{doc_type}"] = 0

    return stats
