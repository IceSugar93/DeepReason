"""构建双层向量索引 — Parent-Child + HyDE 架构

Step 2.3 主脚本（升级版）。执行流程:
1. 加载 data/processed/parent_chunks.json + child_chunks.json
2. 用 BGE-M3 为 Child Chunks 生成稠密向量（精准检索层）
3. 在 Parent Chunks 上构建 BM25 稀疏索引（关键词检索层）
4. 将 Child Chunks + 向量写入 Milvus
5. 验证检索链路: HyDE → Child 检索 → Parent 扩展 → BM25 → RRF → Rerank

使用方式:
    python scripts/build_index.py                      # 首次构建
    python scripts/build_index.py --skip-embeddings    # 跳过 embedding 生成（已缓存）
    python scripts/build_index.py --drop-existing      # 删除已有 Collection 重建
    python scripts/build_index.py --no-hyde            # 验证时不使用 HyDE
"""

import argparse
import json
import os
import sys
import time

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table

from config.settings import (
    CHILD_CHUNKS_FILE,
    PARENT_CHUNKS_FILE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    MILVUS_COLLECTION,
    BM25_INDEX_DIR,
    DENSE_DIM,
)
from src.retrieval.vector_store import (
    connect_milvus,
    disconnect_milvus,
    create_collection,
    insert_embeddings,
    search_dense,
    expand_children_to_parents,
    load_parents,
    get_collection_stats,
)
from src.retrieval.bm25_store import BM25Store
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_retriever import HybridRetriever

console = Console()


# ============================================================================
# Embedding 生成（仅 Child Chunks）
# ============================================================================

def generate_child_embeddings(
    children: list[dict],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[np.ndarray]:
    """用 BGE-M3 为所有 Child Chunks 生成稠密向量。

    Child 层级的文本更短（200-650 chars），embedding 生成更快，
    且小文本的语义表示更精准。

    支持增量断点保存——每 20 个 batch 存一次，中途崩溃可续跑。
    """
    console.print(f"[bold cyan]加载 Embedding 模型[/]: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["content"] for c in children]
    total = len(texts)
    num_batches = (total + batch_size - 1) // batch_size

    # 检查是否有断点可恢复
    cache_path = CHILD_CHUNKS_FILE.replace(".json", "_embeddings.npy")
    checkpoint_path = CHILD_CHUNKS_FILE.replace(".json", "_embeddings_checkpoint.npz")

    all_embeddings: list[np.ndarray] = []
    start_batch = 0

    if os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        all_embeddings = [ckpt[f"e_{j}"].astype(np.float32) for j in range(len(ckpt.files) - 1)]
        start_batch = int(ckpt["batch_idx"])
        console.print(
            f"[yellow]🔄 从断点恢复[/]: 已完成 {len(all_embeddings)} 条 "
            f"(batch {start_batch}/{num_batches})"
        )
    elif os.path.exists(cache_path):
        console.print(f"[yellow]🔄 从完整缓存加载（--skip-embeddings）[/]")
        return load_cached_child_embeddings()

    console.print(
        f"[bold cyan]生成 Child Chunks 稠密向量[/]: "
        f"{total} 条, batch_size={batch_size}, "
        f"共 {num_batches} 批 (纯CPU预计 {total//3//60}-{total//2//60} 分钟)"
    )

    start_time = time.time()
    checkpoint_interval = 20  # 每 20 批保存一次断点

    for batch_idx in range(start_batch, num_batches):
        i = batch_idx * batch_size
        batch = texts[i : i + batch_size]
        batch_num = batch_idx + 1

        # 进度提示（encode 之前打印，避免用户以为卡死）
        if batch_idx == start_batch:
            console.print(
                f"  [bold yellow]⏳ 正在处理[/] batch {batch_num}/{num_batches} "
                f"({len(batch)} 条) — 首个 batch 最慢（模型预热）..."
            )
        else:
            console.print(
                f"  [bold yellow]⏳ 正在处理[/] batch {batch_num}/{num_batches} "
                f"({len(batch)} 条)..."
            )

        # 每次只传一个 batch，避免 Windows 下 DataLoader 多进程 hang
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_embeddings.extend([emb.astype(np.float32) for emb in embeddings])

        # 每批完成后打印统计
        done = i + len(batch)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        console.print(
            f"  [green]✓ batch {batch_num}/{num_batches} 完成[/] — "
            f"{done}/{total} ({done/total*100:.1f}%), "
            f"速度: {rate:.0f} child/s, "
            f"预计剩余: {remaining/60:.0f}min"
        )

        # 增量断点保存
        if (batch_num % checkpoint_interval == 0) or (batch_num == num_batches):
            ckpt_data = {f"e_{j}": all_embeddings[j] for j in range(len(all_embeddings))}
            ckpt_data["batch_idx"] = batch_num  # 已完成到第几个 batch
            np.savez_compressed(checkpoint_path, **ckpt_data)
            console.print(
                f"  [dim]💾 断点已保存 ({len(all_embeddings)} 条)[/]"
            )

    elapsed = time.time() - start_time
    console.print(
        f"[green]✅ Child 稠密向量生成完成[/]: "
        f"{len(all_embeddings)} 条, 耗时 {elapsed/60:.1f}min, "
        f"速度: {total/elapsed:.0f} child/s, "
        f"向量维度: {all_embeddings[0].shape[0]}"
    )

    # 最终缓存到磁盘
    embeddings_array = np.stack(all_embeddings)
    np.save(cache_path, embeddings_array)
    console.print(f"[dim]向量已缓存到: {cache_path}")

    # 清理断点文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return all_embeddings


def load_cached_child_embeddings() -> list[np.ndarray] | None:
    """尝试从缓存加载已生成的 child 向量。"""
    cache_path = CHILD_CHUNKS_FILE.replace(".json", "_embeddings.npy")
    if os.path.exists(cache_path):
        console.print(f"[dim]从缓存加载向量: {cache_path}")
        embeddings_array = np.load(cache_path)
        return [embeddings_array[i] for i in range(embeddings_array.shape[0])]
    return None


# ============================================================================
# BM25 索引（在 Parent 层级）
# ============================================================================

def build_parent_bm25(parents: list[dict]) -> BM25Store:
    """在 Parent Chunks 上构建 BM25 稀疏索引。

    Parent 层级文本完整（1500-3000 chars），关键词信息丰富，
    BM25 的词频统计更有意义。
    """
    console.print(f"[bold cyan]构建 BM25 索引[/]: {len(parents)} 条 Parent Chunks")

    start_time = time.time()
    bm25 = BM25Store()
    bm25.build_index(parents)
    bm25.save(BM25_INDEX_DIR)

    elapsed = time.time() - start_time
    console.print(
        f"[green]✅ BM25 索引构建完成[/]: "
        f"{bm25.num_docs} 条文档, "
        f"词表大小: {len(bm25.idf)}, "
        f"平均长度: {bm25.avgdl:.1f} tokens, "
        f"耗时 {elapsed:.1f}s"
    )
    console.print(f"[dim]BM25 索引已保存到: {BM25_INDEX_DIR}")

    return bm25


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeepReason 双层向量索引构建（Parent-Child + HyDE）"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="跳过 Child embedding 生成（从缓存加载）",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="删除已有 Milvus Collection 后重建",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default="什么是MCP协议？它和Function Calling有什么区别？",
        help="构建完成后用于验证的测试查询",
    )
    parser.add_argument(
        "--no-hyde",
        action="store_true",
        help="验证时禁用 HyDE（做对比测试）",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Phase 0: 启动信息
    # ------------------------------------------------------------------
    console.print()
    console.rule("[bold blue]DeepReason 双层向量索引构建（Parent-Child + HyDE）")
    console.print(f"Milvus Collection: {MILVUS_COLLECTION} (仅 Child Chunks)")
    console.print(f"Embedding 模型: {EMBEDDING_MODEL_NAME}")
    console.print(f"HyDE: {'启用' if not args.no_hyde else '禁用 (对比模式)'}")

    # ------------------------------------------------------------------
    # Phase 1: 加载数据
    # ------------------------------------------------------------------
    console.print()
    console.rule("[bold]Phase 1: 加载 Parent + Child Chunks")

    for path, label in [
        (PARENT_CHUNKS_FILE, "Parent Chunks"),
        (CHILD_CHUNKS_FILE, "Child Chunks"),
    ]:
        if not os.path.exists(path):
            console.print(f"[red]错误[/]: {label} 文件不存在 — {path}")
            console.print("请先运行 python src/utils/parsers.py 生成 chunk")
            sys.exit(1)

    with open(PARENT_CHUNKS_FILE, "r", encoding="utf-8") as f:
        parents = json.load(f)
    with open(CHILD_CHUNKS_FILE, "r", encoding="utf-8") as f:
        children = json.load(f)

    console.print(f"Parent Chunks: {len(parents):5d} 条")
    console.print(f"Child  Chunks: {len(children):5d} 条")
    console.print(f"Child/Parent 比: {len(children)/max(len(parents),1):.1f}:1")

    # 加载 parents 到内存缓存（供后续 lookup_parents 使用）
    load_parents(PARENT_CHUNKS_FILE)
    console.print(f"[dim]Parent 缓存已加载: {len(parents)} 条")

    # ------------------------------------------------------------------
    # Phase 2: 生成 Child 稠密向量
    # ------------------------------------------------------------------
    console.print()
    console.rule("[bold]Phase 2: Child Chunks 稠密向量 (BGE-M3)")

    if args.skip_embeddings:
        embeddings = load_cached_child_embeddings()
        if embeddings is None:
            console.print("[red]未找到缓存向量，回退到重新生成[/]")
            embeddings = generate_child_embeddings(children)
    else:
        embeddings = generate_child_embeddings(children)

    if len(embeddings) != len(children):
        console.print(
            f"[red]错误[/]: 向量数 ({len(embeddings)}) "
            f"与 child 数 ({len(children)}) 不匹配"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 3: BM25 索引（在 Parent 层级）
    # ------------------------------------------------------------------
    console.print()
    console.rule("[bold]Phase 3: Parent Chunks BM25 稀疏索引")

    bm25 = build_parent_bm25(parents)

    # ------------------------------------------------------------------
    # Phase 4: Milvus 向量入库（仅 Child Chunks）
    # ------------------------------------------------------------------
    console.print()
    console.rule("[bold]Phase 4: Child Chunks → Milvus 向量入库")

    connect_milvus()
    col = create_collection(drop_if_exists=args.drop_existing)
    console.print(f"Milvus Collection '{MILVUS_COLLECTION}' 已就绪")

    col.flush()
    if col.num_entities > 0 and not args.drop_existing:
        console.print(
            f"[yellow]⚠ Collection 已有 {col.num_entities} 条数据，跳过插入[/]"
        )
        console.print("如需重建，请使用 --drop-existing 参数")
    else:
        console.print(f"开始插入: {len(children)} 条 Child Chunks + 向量...")
        start_time = time.time()
        inserted = insert_embeddings(children, embeddings)
        elapsed = time.time() - start_time
        console.print(
            f"[green]✅ 插入完成[/]: {inserted} 条, 耗时 {elapsed:.1f}s"
        )

    # ------------------------------------------------------------------
    # Phase 5: 检索验证
    # ------------------------------------------------------------------
    console.print()
    console.rule("[bold]Phase 5: 检索链路验证")

    # 初始化混合检索器（集成 HyDE + Parent-Child + BM25 + Rerank）
    reranker = Reranker()
    retriever = HybridRetriever(
        bm25_store=bm25,
        reranker=reranker,
        enable_hyde=not args.no_hyde,
    )

    # 生成查询向量（不用 HyDE 时的备选）
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = emb_model.encode(
        args.test_query,
        normalize_embeddings=True,
    ).astype(np.float32)

    console.print(f"测试查询: [bold]\"{args.test_query}\"[/]")
    console.print(f"HyDE: {'[green]启用[/]' if not args.no_hyde else '[yellow]禁用[/]'}")

    # 分别测试三种检索方式做对比
    # (a) 纯稠密检索 (Child → Parent 扩展)
    t0 = time.time()
    child_hits = search_dense(query_embedding, top_k=10)
    dense_parents = expand_children_to_parents(child_hits, threshold=2)
    dense_time = time.time() - t0

    # (b) 纯 BM25 检索（Parent 层级）
    t0 = time.time()
    sparse_hits = bm25.search(args.test_query, top_k=5)
    sparse_time = time.time() - t0

    # (c) 完整混合检索（HyDE + Parent-Child + BM25 + Rerank）
    t0 = time.time()
    hybrid_hits = retriever.search(
        args.test_query,
        query_embedding,
        top_k=5,
    )
    hybrid_time = time.time() - t0

    # 打印结果对比表
    _print_search_results(
        "Dense (Child→Parent 扩展)", dense_parents[:5], dense_time,
    )
    _print_search_results("BM25 (Parent 层级)", sparse_hits, sparse_time)
    _print_search_results(
        f"混合检索 + HyDE + Rerank (完整链路)", hybrid_hits, hybrid_time,
    )

    # 统计
    console.print()
    stats = get_collection_stats()
    console.print(
        f"[bold]检索架构统计:[/] "
        f"{stats['total_parents']} Parents / {stats['total_children']} Children "
        f"(Child/Parent: {stats['child_parent_ratio']})"
    )

    disconnect_milvus()
    console.print()
    console.rule("[bold green]Step 2.3 完成！高级 RAG 检索链路已验证通过。")
    console.print()
    console.print("[bold]检索链路:[/]")
    console.print("  查询 → [cyan]HyDE 假设答案[/] → embed")
    console.print("       → [cyan]Child Chunk 精准检索[/] (Milvus)")
    console.print("       → [cyan]Parent 扩展[/] (Small-to-Big)")
    console.print("       → + BM25 关键词 (Parent 层级)")
    console.print("       → RRF 融合 → Cross-Encoder 重排序")
    console.print("       → [green]Top-K Parent Chunks[/]")
    console.print()
    console.print("接下来: [bold]Step 3 — LangGraph 状态机核心搭建[/]")


# ============================================================================
# 结果展示
# ============================================================================

def _print_search_results(label: str, hits: list[dict], elapsed: float) -> None:
    """用 Rich Table 打印检索结果。"""
    console.print()
    console.print(f"[bold cyan]{label}[/] ({len(hits)} 条, {elapsed:.2f}s)")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=3, justify="right")
    table.add_column("Score", width=8)
    table.add_column("Level", width=7)
    table.add_column("C.Count", width=7)
    table.add_column("Type", width=7)
    table.add_column("Title", width=35)
    table.add_column("Content (前60字)", width=55)

    for i, hit in enumerate(hits, 1):
        score = f"{hit.get('rerank_score', hit['score']):.4f}"
        level = hit.get("chunk_level", "parent")[:6]
        child_cnt = str(hit.get("child_count", "-"))
        doc_type = hit.get("doc_type", "")
        title = hit.get("title", "")
        if len(title) > 32:
            title = title[:29] + "..."
        content = hit["content"][:57].replace("\n", " ") + "..."
        if len(hit["content"]) > 60:
            content = hit["content"][:57].replace("\n", " ") + "..."

        table.add_row(str(i), score, level, child_cnt, doc_type, title, content)

    console.print(table)


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    main()
