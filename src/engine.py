"""DeepReason 推理引擎 — 顶层入口

- DeepReasonEngine: 初始化检索器 + Graph，暴露 run(query) 接口
- 通过 contextvars 向图节点注入不可序列化的共享资源（Retriever、Embedding 模型），
  确保 LangGraph checkpoint 机制正常工作。

架构 (2026-07-24): 3-Agent（Generator/Critic/Reviser）+ 检索链（Planner）。
"""

import json
import time
import uuid
import contextvars
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL_NAME
from src.retrieval.hybrid_retriever import HybridRetriever
from src.graph import build_graph, set_retriever_context
from src.graph.state import AgentState


class DeepReasonEngine:
    """DeepReason 3-Agent 审查-修订推理引擎。

    生命周期:
        engine = DeepReasonEngine(retriever, embed_model)
        result = engine.run("什么是 RAG？")
        print(result["answer"])
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        embed_model: Optional[SentenceTransformer] = None,
    ):
        self.retriever = retriever
        self.embed_model = embed_model or SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.graph = build_graph()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def run(self, query: str, **kwargs) -> dict:
        """执行完整的推理流程。

        Returns:
            {
                "query", "answer", "confidence", "verdict",
                "review_summary": {"total_rounds", "revision_rounds", ...},
                "sources", "internal_contexts", "retrieval_stats",
                "trace_id", "wall_time_ms",
            }
        """
        set_retriever_context(self.retriever, self.embed_model)

        trace_id = str(uuid.uuid4())[:8]
        initial_state: dict = {
            "query": query,
            "retrieved_docs": [],
            "sub_question_docs": {},
            "complexity": "simple",
            "sub_questions": [],
            "draft_answer": "",
            "review_round": 0,
            "critic_ruling": {},
            "review_history": [],
            "revision_round": 0,
            "refined_answer": "",
            "previous_issues": [],
            "revision_history": [],
            "verify_result": {},
            "final_answer": "",
            "confidence": 0.0,
            "retrieval_hops": 0,
            "errors": [],
        }
        initial_state.update(kwargs)

        start = time.time()
        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return {
                "query": query,
                "answer": f"推理引擎执行出错: {e}",
                "confidence": 0.0,
                "verdict": "error",
                "review_summary": {},
                "sources": [],
                "internal_contexts": [],
                "retrieval_stats": {},
                "trace_id": trace_id,
                "wall_time_ms": elapsed_ms,
                "error": str(e),
            }

        elapsed_ms = int((time.time() - start) * 1000)

        docs = final_state.get("retrieved_docs", [])
        critic_ruling = final_state.get("critic_ruling", {})

        return {
            "query": query,
            "answer": (
                final_state.get("final_answer")
                or final_state.get("refined_answer")
                or final_state.get("draft_answer", "")
            ),
            "confidence": final_state.get("confidence", 0.0),
            "verdict": critic_ruling.get("verdict", "unknown"),
            "review_summary": {
                "total_rounds": final_state.get("review_round", 0),
                "revision_rounds": final_state.get("revision_round", 0),
                "critic_confidence": critic_ruling.get("confidence", 0.0),
                "issues_found": len(critic_ruling.get("issues", [])),
            },
            "sources": [
                {
                    "chunk_id": d.get("chunk_id", ""),
                    "title": d.get("title", "未知来源"),
                    "relevance": round(d.get("score", 0), 4),
                }
                for d in docs[:8]
            ],
            "internal_contexts": [d.get("content", "") for d in docs],
            "retrieval_stats": {
                "hops": final_state.get("retrieval_hops", 0),
                "total_docs": len(docs),
                "hyde_used": True,
            },
            "trace_id": trace_id,
            "wall_time_ms": elapsed_ms,
            "state": {
                k: v
                for k, v in final_state.items()
                if k not in ("retrieved_docs", "errors")
            },
        }

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def run_batch(self, queries: list[str]) -> list[dict]:
        return [self.run(q) for q in queries]

    def stream(self, query: str):
        """流式执行推理图，每次 yield 一个节点的事件。"""
        set_retriever_context(self.retriever, self.embed_model)

        initial_state: dict = {
            "query": query,
            "retrieved_docs": [],
            "sub_question_docs": {},
            "complexity": "simple",
            "sub_questions": [],
            "draft_answer": "",
            "review_round": 0,
            "critic_ruling": {},
            "review_history": [],
            "revision_round": 0,
            "refined_answer": "",
            "previous_issues": [],
            "revision_history": [],
            "verify_result": {},
            "final_answer": "",
            "confidence": 0.0,
            "retrieval_hops": 0,
            "errors": [],
        }

        for event in self.graph.stream(initial_state):
            yield event
