"""检索质量快速验证 — 确认 Step 2.3 产出可用后再进入 Step 3"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL_NAME, PARENT_CHUNKS_FILE
from src.retrieval.vector_store import (
    connect_milvus, search_dense, expand_children_to_parents, load_parents, disconnect_milvus,
)
from src.retrieval.bm25_store import BM25Store
from config.settings import BM25_INDEX_DIR

load_parents(PARENT_CHUNKS_FILE)
emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
bm25 = BM25Store()
bm25.load(BM25_INDEX_DIR)
connect_milvus()

test_queries = [
    "什么是MCP协议？它和Function Calling有什么区别？",
    "RAG系统中如何评估检索质量？",
    "Multi-Agent辩论机制如何提高推理准确性？",
    "LLM Agent的工具使用能力是如何演进的？",
]

for query in test_queries:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"查询: {query}")
    print(sep)

    # 稠密检索 + Parent 扩展
    q_emb = emb_model.encode(query, normalize_embeddings=True)
    child_hits = search_dense(q_emb, top_k=10)
    dense_parents = expand_children_to_parents(child_hits, threshold=2)

    # BM25 检索
    sparse_hits = bm25.search(query, top_k=5)

    print("\n--- 稠密检索 (Child→Parent扩展) Top-3 ---")
    for i, hit in enumerate(dense_parents[:3]):
        title = hit.get("title", "?")[:55]
        score = hit.get("score", 0)
        level = hit.get("chunk_level", "?")
        content = hit["content"][:120].replace("\n", " ")
        print(f"  [{i+1}] score={score:.4f} level={level} | {title}")
        print(f"      {content}...")

    print("\n--- BM25 关键词检索 Top-3 ---")
    for i, hit in enumerate(sparse_hits[:3]):
        title = hit.get("title", "?")[:55]
        score = hit.get("score", 0)
        content = hit["content"][:120].replace("\n", " ")
        print(f"  [{i+1}] score={score:.4f} | {title}")
        print(f"      {content}...")

disconnect_milvus()
print("\n✅ 检索验证完成")
