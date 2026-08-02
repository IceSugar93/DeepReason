"""DeepReason Web 界面 — FastAPI 后端

提供三类能力：
1. GET /api/stream?query=...  — SSE 流式推理：图节点执行过程实时推送（辩论可视化）
2. GET /api/query            — 同步完整推理（POST {"query": ...}）
3. GET /api/eval[/{name}]    — 历史 eval 结果列表 / 逐题明细

启动方式（项目根目录）:
    uvicorn web.app:app --port 8000
"""

import asyncio
import io
import json
import logging
import os
import queue
import re
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# 保证从任意 cwd 都能导入 config/src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 终端编码：引擎节点内的 rich 控制台输出含 emoji，stdout 默认 GBK 会炸线程
# （与 eval/run_eval_debate.py 的同样处理保持一致）
if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr is not None and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config.settings import (
    BM25_INDEX_DIR,
    EMBEDDING_MODEL_NAME,
    FINAL_TOP_K,
    GENERATOR_MODEL,
    PARENT_CHUNKS_FILE,
)
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import connect_milvus, disconnect_milvus, load_parents

logger = logging.getLogger("deepreason.web")

WEB_ROOT = Path(__file__).parent
STATIC_DIR = WEB_ROOT / "static"
EVAL_DIR = WEB_ROOT.parent / "eval" / "results"

# 全局引擎单例（lifespan 中初始化；BGE-M3 只加载一次）
_ENGINE = None
_INIT_ERROR: str | None = None


# ============================================================================
# 引擎初始化（lifespan）
# ============================================================================

def _init_engine():
    """初始化检索组件 + 推理引擎，与 eval 脚本保持同一套装配。"""
    from src.engine import DeepReasonEngine
    from sentence_transformers import SentenceTransformer

    load_parents(PARENT_CHUNKS_FILE)
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    bm25 = BM25Store()
    bm25.load(BM25_INDEX_DIR)
    reranker = Reranker()
    connect_milvus()
    retriever = HybridRetriever(
        bm25_store=bm25,
        reranker=reranker,
        enable_hyde=True,
    )
    return DeepReasonEngine(retriever, emb_model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ENGINE, _INIT_ERROR
    _ENGINE, _INIT_ERROR = None, None
    logger.info("初始化引擎组件（BGE-M3 + Milvus + BM25 + Reranker）...")
    try:
        _ENGINE = _init_engine()
        logger.info(f"引擎就绪，生成模型: {GENERATOR_MODEL}")
    except Exception as e:  # Milvus 未启动 / 索引缺失等 → 应用仍可启动，接口报 503
        _INIT_ERROR = f"{type(e).__name__}: {e}"
        logger.error("引擎初始化失败: %s", _INIT_ERROR)
    yield
    try:
        disconnect_milvus()
    except Exception:
        pass


app = FastAPI(title="DeepReason", lifespan=lifespan)


# ============================================================================
# 工具函数
# ============================================================================

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _compact_docs(docs) -> list[dict]:
    """裁剪检索文档：只保留元信息，去掉 content 大文本。"""
    out = []
    for d in docs[:10]:
        out.append(
            {
                "chunk_id": d.get("chunk_id", ""),
                "title": d.get("title", "未知来源"),
                "doc_type": d.get("doc_type", ""),
                "score": round(d.get("score", 0), 4),
            }
        )
    return out


def _compact_node(node: str, payload: dict) -> dict:
    """把单个节点返回的 state 更新裁剪为前端展示友好的结构。"""
    p = dict(payload)
    if "retrieved_docs" in p:
        p["retrieved_docs"] = _compact_docs(p["retrieved_docs"])
    if "sub_question_docs" in p:
        p["sub_question_docs"] = {
            k: _compact_docs(v) for k, v in p["sub_question_docs"].items()
        }
    if isinstance(p.get("review_history"), list):
        # 累加器字段：payload 里是本次新增条目，只需最新一条
        p["review_history"] = p["review_history"][-1:]
    if isinstance(p.get("revision_history"), list):
        p["revision_history"] = p["revision_history"][-1:]
    return p


def _build_result(query: str, state: dict) -> dict:
    """由最终累积 state 构造 engine.run 风格的返回结构。"""
    critic_ruling = state.get("critic_ruling", {})
    docs = state.get("retrieved_docs", [])
    return {
        "query": query,
        "answer": (
            state.get("final_answer")
            or state.get("refined_answer")
            or state.get("draft_answer", "")
        ),
        "confidence": state.get("confidence", 0.0),
        "verdict": critic_ruling.get("verdict", "unknown"),
        "review_summary": {
            "total_rounds": state.get("review_round", 0),
            "revision_rounds": state.get("revision_round", 0),
            "critic_confidence": critic_ruling.get("confidence", 0.0),
            "issues_found": len(critic_ruling.get("issues", [])),
            "converged": state.get("converged", False),
            "answer_similarity": state.get("answer_similarity", None),
        },
        "retrieval_stats": {
            "hops": state.get("retrieval_hops", 0),
            "total_docs": len(docs),
        },
        "internal_contexts": [d.get("content", "") for d in docs],
        "answer_annotations": state.get("answer_annotations", {}),
    }


# ============================================================================
# 健康检查
# ============================================================================

@app.get("/api/health")
def health():
    return {
        "ready": _ENGINE is not None,
        "error": _INIT_ERROR,
        "generator_model": GENERATOR_MODEL,
        "top_k": FINAL_TOP_K,
    }


# ============================================================================
# 推理接口
# ============================================================================

@app.post("/api/query")
def run_query(payload: dict):
    if _ENGINE is None:
        raise HTTPException(503, f"引擎未就绪: {_INIT_ERROR}")
    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query 不能为空")
    if len(query) > 500:
        raise HTTPException(400, "query 过长（≤500 字符）")
    result = _ENGINE.run(query)
    result.pop("state", None)  # 同步接口只返回结果摘要
    return result


@app.get("/api/stream")
async def stream(query: str):
    """SSE 流式推理：图每个节点完成后推送一个事件。"""
    if not query.strip():
        raise HTTPException(400, "query 不能为空")
    if len(query) > 500:
        raise HTTPException(400, "query 过长（≤500 字符）")
    if _ENGINE is None:
        raise HTTPException(503, f"引擎未就绪: {_INIT_ERROR}")

    q: "queue.Queue" = queue.Queue()

    def worker():
        try:
            state: dict = {}
            # engine.stream 内部已 set_retriever_context + 构造 initial_state
            for event in _ENGINE.stream(query):
                for node, payload in event.items():
                    state.update(payload)
                    q.put(("node", node, payload))
            q.put(("done", state))
        except Exception as e:
            logger.exception("流式推理失败")
            q.put(("error", str(e)))

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        while True:
            item = await asyncio.to_thread(q.get)
            kind = item[0]
            if kind == "node":
                yield _sse({"type": "node", "node": item[1], "data": _compact_node(item[1], item[2])})
            elif kind == "done":
                yield _sse({"type": "done", "result": _build_result(query, item[1])})
                break
            else:
                yield _sse({"type": "error", "message": str(item[1])})
                break

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# Eval 结果接口
# ============================================================================

_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


def _eval_files():
    if not EVAL_DIR.exists():
        return []
    return sorted(EVAL_DIR.glob("eval_debate_three_way_*.json"), reverse=True)


@app.get("/api/eval")
def list_eval_runs():
    runs = []
    for f in _eval_files():
        m = _TIMESTAMP_RE.search(f.name)
        timestamp = f"{m.group(1)} {m.group(2)}" if m else f.stem
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        runs.append(
            {
                "filename": f.name,
                "timestamp": timestamp,
                "config": d.get("config", {}),
                "summary": d.get("summary", {}),
            }
        )
    return runs


@app.get("/api/eval/{filename}")
def get_eval_run(filename: str):
    if not _TIMESTAMP_RE.search(filename) or not filename.endswith(".json"):
        raise HTTPException(400, "非法文件名")
    f = EVAL_DIR / filename
    if not f.exists():
        raise HTTPException(404, f"运行不存在: {filename}")
    return json.loads(f.read_text(encoding="utf-8"))


# ============================================================================
# 静态页面
# ============================================================================

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
