"""HyDE 查询扩展 — 用假设性答案桥接查询-文档语义鸿沟

核心思路 (Gao et al., 2023):
  短查询和长文档之间存在语义鸿沟。LLM 先生成一个"假设性理想答案"，
  再用这个假设答案去做 embedding 检索——因为编出来的答案在语义空间里
  更接近真实文档。

DeepReason 特化:
  Multi-HyDE: 不同 Agent（Advocate/Skeptic）从不同视角生成假设答案，
  各自检索后通过 RRF 融合，在检索阶段就引入对抗性视角。
"""

from openai import OpenAI

from config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    HYDE_MODEL,
    HYDE_TEMPERATURE,
    MULTI_HYDE_PERSPECTIVES,
)


# ============================================================================
# HyDE 提示词模板
# ============================================================================

# 基础 HyDE: 生成一个中性的假设性答案
HYDE_SYSTEM_PROMPT = """你是一个 AI 知识助手。用户会提出一个技术问题，你的任务是
写出一段"假设性的理想答案"。这段答案不要求事实完全准确——它的目的是捕捉正确答案
的语义特征和写作风格，用于后续向量检索。

请用一段话（80-150 字）写出这个假设答案。使用技术文档的风格，包含可能的关键术语。
不要写"我不知道"或承认不确定性——请自信地写出你猜测的答案。"""

# Multi-HyDE: 不同视角的提示词变体
PERSPECTIVE_PROMPTS = {
    "neutral": "请用中立的、百科全书式的语气写出假设答案。",
    "supportive": "请用支持/论证的语气写出假设答案——就像在为某个观点寻找证据支持。",
    "critical": "请用批判/反驳的语气写出假设答案——关注潜在的局限性和反例。",
}

# ============================================================================
# HyDE 核心类
# ============================================================================

class HyDE:
    """HyDE 查询扩展器。

    在检索前用 LLM 生成假设性答案，将短查询扩展为语义丰富的文档级表示。

    使用示例:
        hyde = HyDE()
        hypothesis = hyde.generate("什么是 MCP 协议？")
        # → "MCP（Model Context Protocol）是一种标准化的通信协议..."
    """

    def __init__(
        self,
        model: str = HYDE_MODEL,
        temperature: float = HYDE_TEMPERATURE,
    ):
        self._model = model
        self._temperature = temperature
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """延迟初始化 OpenAI 客户端。"""
        if self._client is None:
            self._client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
            )
        return self._client

    # ------------------------------------------------------------------
    # 基础 HyDE: 单视角假设答案
    # ------------------------------------------------------------------

    def generate(self, query: str, perspective: str = "neutral") -> str:
        """为查询生成一个假设性答案。

        Args:
            query: 用户原始查询。
            perspective: 视角，可选 "neutral"、"supportive"、"critical"。

        Returns:
            假设性答案文本（80-150 字）。
        """
        perspective_instruction = PERSPECTIVE_PROMPTS.get(
            perspective, PERSPECTIVE_PROMPTS["neutral"]
        )

        response = self.client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": f"{perspective_instruction}\n\n查询: {query}"},
            ],
            temperature=self._temperature,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Multi-HyDE: 多视角假设答案生成
    # ------------------------------------------------------------------

    def generate_multi(
        self,
        query: str,
        perspectives: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """从多个视角生成假设答案。

        Args:
            query: 用户原始查询。
            perspectives: 视角列表，默认使用 MULTI_HYDE_PERSPECTIVES。

        Returns:
            [(perspective_label, hypothesis_text), ...]
            例如 [("neutral", "..."), ("supportive", "..."), ("critical", "...")]
        """
        perspectives = perspectives or MULTI_HYDE_PERSPECTIVES

        results = []
        for p in perspectives:
            hypothesis = self.generate(query, perspective=p)
            results.append((p, hypothesis))

        return results

    # ------------------------------------------------------------------
    # 便捷方法：生成用于检索的扩展查询文本
    # ------------------------------------------------------------------

    def expand_query(self, query: str) -> str:
        """生成中性假设答案，拼接在原始查询后面用于检索。

        这是最简单的 HyDE 用法——把假设答案当检索文本。

        Args:
            query: 用户原始查询。

        Returns:
            原始查询 + 假设答案拼接后的文本。
        """
        hypothesis = self.generate(query, perspective="neutral")
        return f"{query}\n\n{hypothesis}"

    def expand_query_multi(
        self,
        query: str,
        perspectives: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """从多个视角生成扩展查询文本。

        每个视角的假设答案独立用于一次检索，结果通过 RRF 融合。
        这比单视角 HyDE 召回率更高（2025 研究：Recall +12-18%）。

        Args:
            query: 用户原始查询。
            perspectives: 视角列表。

        Returns:
            [(perspective_label, expanded_query_text), ...]
        """
        perspectives = perspectives or MULTI_HYDE_PERSPECTIVES

        results = []
        for p in perspectives:
            hypothesis = self.generate(query, perspective=p)
            expanded = f"{query}\n\n{hypothesis}"
            results.append((p, expanded))

        return results
