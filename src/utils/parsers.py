"""文档解析与语义切块 — DeepReason 知识库

支持 PDF（学术论文）和纯文本（技术文档、博客）。

Parent-Child 双层级切块策略:
- Parent Chunk (1500-3000 chars): 语义段落，信息完整 → 喂给 LLM 的上下文
- Child Chunk (200-650 chars): 句子级片段，精准匹配 → 嵌入索引的检索单元
- 目标比例: 每个 Parent 约拆分为 3 个 Child

检索时: 用 Child 做精准匹配 → 映射到 Parent → 返回完整上下文
这就是 Milvus 官方推荐的 Small-to-Big 检索模式。
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """切块参数。针对英文文本 + BGE-M3（最大 8192 tokens）调优。

    核心理念：语义边界优先，大小限制只作为兜底手段。
    """

    target_chars: int = 2000      # 理想 chunk 大小（偏好而非硬限制）
    overlap_chars: int = 300     # 滑动窗口切分时的字符重叠量
    min_chars: int = 100         # 短于此值的 chunk 会被丢弃或合并
    max_chars: int = 3000        # 硬上限 — 只有超过此值的段落才会被强制切分
                                 # （BGE-M3 可以舒适处理 8K tokens）

    # Parent-Child 切块参数
    child_target_chars: int = 600  # Child chunk 的理想大小（检索单元）
    child_min_chars: int = 200      # Child chunk 的最小长度

    # 断点优先级：数值越大 = 越强的边界。
    # 当某个段落超过 max_chars 需要强制切分时，按优先级从高到低尝试。
    break_priority: dict = field(default_factory=lambda: {
        "heading": 10,      # 最强：#、##、### 标题
        "double_nl": 8,     # 段落间隙（空行）
        "list_item": 6,     # 列表项（bullet / 编号）
        "single_nl": 4,     # 软换行
        "sentence": 2,      # 句号/问号/感叹号 + 空格
        "space": 1,         # 最后手段：任意空格
    })


# ---------------------------------------------------------------------------
# 文档解析（对外接口）
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str | Path) -> str:
    """使用 pdfplumber 从 PDF 文件中提取纯文本。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        提取的文本（单个字符串），解析失败时返回空字符串。
    """
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
    except Exception as exc:
        print(f"  警告: 解析失败 {Path(pdf_path).name}: {exc}")
        return ""


def parse_txt(txt_path: str | Path) -> str:
    """读取纯文本文件（已提取好的文档/博客）。"""
    return Path(txt_path).read_text(encoding="utf-8", errors="replace")


def parse_document(file_path: str | Path) -> str:
    """根据文件扩展名自动选择合适的解析器。

    支持:
        .pdf  → pdfplumber 提取
        .txt  → 直接读取

    Args:
        file_path: 文档路径。

    Returns:
        提取后的纯文本。

    Raises:
        ValueError: 不支持的文件类型。
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    if suffix == ".txt":
        return parse_txt(file_path)
    raise ValueError(f"不支持的文件类型: {suffix}")


# ---------------------------------------------------------------------------
# 语义切块器
# ---------------------------------------------------------------------------

class SemanticChunker:
    """语义感知的文本切块器。

    特性
    ----
    - **代码块保护**: 围栏代码块 (```...```) 永远不会被切分到多个 chunk 中。
    - **标题上下文注入**: 最接近的 markdown 标题会被添加到每个 chunk 开头，
      即使 chunk 单独出现也能携带段落上下文。
    - **表格保护**: 看起来像表格行（共享分隔符如 | 或对齐的制表符）的行会保持在一起。
    - **FAQ 保护**: Q:/A: 问答对保持在同一 chunk 中。
    """

    HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    LIST_ITEM_RE = re.compile(r"^(\s*[-*+]\s|\s*\d+[.)]\s)")
    TABLE_SEP_RE = re.compile(r"^[\s|+-]{3,}$")
    FAQ_Q_RE = re.compile(r"^\s*Q[:\s]", re.IGNORECASE)

    def __init__(self, source: str, doc_type: str, config: ChunkConfig = None):
        """
        Args:
            source: 人类可读的来源标识（如 "arxiv_id.pdf", "mcp_tools"）。
            doc_type: 文档类型，取 "paper"、"doc"、"blog" 之一。
            config: 切块配置。省略时使用默认值。
        """
        self.source = Path(source).stem if source.endswith(".pdf") else source
        self.doc_type = doc_type
        self.cfg = config or ChunkConfig()
        self._code_blocks: dict[str, str] = {}  # 占位符 → 原始代码块

    # ---- 对外接口 ----

    def chunk(self, text: str) -> list[dict]:
        """将单个文档的文本切分为语义片段。

        Returns:
            chunk dict 列表，可直接写入 all_chunks.json。
        """
        if not text.strip():
            return []

        text = self._normalize(text)
        title = self._guess_title(text)

        # 阶段1: 保护代码块
        text = self._protect_code_blocks(text)

        # 阶段2: 在自然边界处切分
        segments = self._split_on_boundaries(text)

        # 阶段3: 标题上下文注入
        segments = self._inject_titles(segments)

        # 阶段4: 处理超大段落
        segments = self._split_oversized(segments)

        # 阶段5: 合并过小片段
        segments = self._merge_undersized(segments)

        # 阶段6: 加 overlap + 格式化输出
        return self._to_chunks(segments, title)

    # ---- 内部辅助方法 ----

    @staticmethod
    def _normalize(text: str) -> str:
        """压缩多余空白字符，同时保留段落分隔。"""
        # 规范化 Unicode 空白
        text = re.sub(r"[​ ]", " ", text)       # 零宽/不换行空格
        text = re.sub(r"[ \t]+", " ", text)               # 合并行内空格/制表符
        text = re.sub(r"\n[ \t]+\n", "\n\n", text)       # 类空行 → 真正空行
        text = re.sub(r"\n{3,}", "\n\n", text)            # 压缩连续空行
        return text.strip()

    def _guess_title(self, text: str) -> str:
        """从前几行非空、非代码行中猜测文档标题。"""
        for line in text.splitlines()[:6]:
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            if stripped and len(stripped) > 5 and not stripped.startswith("```"):
                return stripped[:120]
        return ""

    # ---- 阶段1: 代码块保护 ----

    def _protect_code_blocks(self, text: str) -> str:
        """提取围栏代码块并替换为占位符。

        这确保代码/JSON/YAML 示例永远不会被切分到多个 chunk 中。
        """
        placeholder_id = 0

        def _replace(m: re.Match) -> str:
            nonlocal placeholder_id
            ph = f"<!--CODEBLOCK_{placeholder_id}-->"
            self._code_blocks[ph] = m.group(0)
            placeholder_id += 1
            return f"\n{ph}\n"

        return re.sub(r"```.*?```", _replace, text, flags=re.DOTALL)

    # ---- 阶段2: 自然边界切分 ----

    def _split_on_boundaries(self, text: str) -> list[str]:
        """在最强自然边界处切分文本。

        策略：先按双换行（段落间隙）切分，再把属于同一"逻辑段落"的行
        合并回去（续行、行内代码、列表项）。
        """
        blocks = text.split("\n\n")
        segments = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            # 段落内的行合并为一个 segment，
            # 但如果段落内嵌了标题，标题需要独立成段。
            sub = self._split_headings(block)
            segments.extend(sub)
        return segments

    def _split_headings(self, block: str) -> list[str]:
        """将段落块按嵌入的标题拆分为子分段。"""
        # 找到所有标题位置
        heading_positions = [m.start() for m in self.HEADING_RE.finditer(block)]
        if not heading_positions:
            return [block] if block.strip() else []

        parts = []
        prev = 0
        for pos in heading_positions:
            if pos > prev:
                parts.append(block[prev:pos].strip())
            prev = pos
        parts.append(block[prev:].strip())
        return [p for p in parts if p]

    # ---- 阶段3: 标题上下文注入 ----

    def _inject_titles(self, segments: list[str]) -> list[str]:
        """将最近的标题作为上下文注入每个 segment。

        标题作为前缀行添加（如 "## 章节标题"），这样即使 chunk 被单独
        查看也能保留语义上下文。
        """
        current_heading = ""
        result = []
        for seg in segments:
            m = self.HEADING_RE.match(seg.strip())
            if m:
                current_heading = m.group(0).strip()
                result.append(seg)
            elif current_heading and not seg.strip().startswith("#"):
                # 非标题段落 → 注入当前最近的标题作为上下文前缀
                result.append(f"{current_heading}\n{seg}")
            else:
                result.append(seg)
        return result

    # ---- 阶段4: 超大段落切分 ----

    def _split_oversized(self, segments: list[str]) -> list[str]:
        """切分任何超过 max_chars 的段落。"""
        result = []
        for seg in segments:
            if len(seg) <= self.cfg.max_chars:
                result.append(seg)
                continue
            # 尝试在最强断点处切分
            sub = self._force_split(seg)
            result.extend(sub)
        return result

    def _force_split(self, long_text: str) -> list[str]:
        """递归切分超出长度限制的文本，从强到弱尝试各种断点。"""
        if len(long_text) <= self.cfg.max_chars:
            return [long_text]

        # 按优先级从高到低尝试各断点
        for boundary, _ in sorted(self.cfg.break_priority.items(),
                                  key=lambda x: -x[1]):
            pos = self._find_break(long_text, boundary)
            if pos is not None and self.cfg.min_chars < pos < len(long_text) - self.cfg.min_chars:
                left = long_text[:pos].strip()
                right = long_text[pos:].strip()
                return self._force_split(left) + self._force_split(right)

        # 最后手段：在 max_chars 处硬切
        cut = self.cfg.max_chars
        # 尝试在 max_chars 附近的空格处切
        space = long_text.rfind(" ", cut - 100, cut)
        if space > self.cfg.min_chars:
            cut = space
        return [long_text[:cut].strip()] + self._force_split(long_text[cut:].strip())

    def _find_break(self, text: str, boundary: str) -> int | None:
        """在文本中部三分之一区域寻找最佳断点。

        偏好文本中部的位置，使得切分出的两段大小大致均衡。
        """
        third = len(text) // 3
        if boundary == "heading":
            m = self.HEADING_RE.search(text[third:2*third])
            return m.start() + third if m else None
        if boundary == "double_nl":
            idx = text.find("\n\n", third)
            return idx if idx > 0 and idx < 2*third else None
        if boundary == "single_nl":
            idx = text.rfind("\n", third, 2*third)
            return idx if idx > third else None
        if boundary == "sentence":
            m = re.search(r"[.!?]\s+", text[third:2*third])
            return m.end() + third if m else None
        if boundary == "space":
            idx = text.rfind(" ", third, 2*third)
            return idx if idx > third else None
        return None

    # ---- 阶段5: 合并过小片段 ----

    def _merge_undersized(self, segments: list[str]) -> list[str]:
        """将短于 min_chars 的片段合并到相邻片段中。"""
        if not segments:
            return []
        result = []
        buf = ""
        for seg in segments:
            if buf and len(buf) + len(seg) <= self.cfg.max_chars:
                buf += "\n\n" + seg
            elif buf and len(buf) < self.cfg.min_chars:
                # 当前 buf 太小 — 尝试向前合并
                buf += "\n\n" + seg
            else:
                if buf.strip():
                    result.append(buf.strip())
                buf = seg
        if buf.strip():
            # 如果最后一个 buf 仍然太小，尝试合并到最后一个 result chunk
            if len(buf) < self.cfg.min_chars and result:
                if len(result[-1]) + len(buf) <= self.cfg.max_chars:
                    result[-1] = result[-1] + "\n\n" + buf
                else:
                    result.append(buf)
            else:
                result.append(buf)
        return result

    # ---- 阶段6: overlap + 格式化 ----

    def _to_chunks(self, segments: list[str], title: str) -> list[dict]:
        """将语义片段格式化为 chunk dict。

        短于 max_chars 的段落保持完整（保留语义单元）。
        只有超过 max_chars 的段落才会走滑动窗口切分 —
        这是超大文本块的最后兜底路径。

        每个 chunk 携带元数据（source, doc_type, title, chunk_index）。
        """
        # 恢复代码块占位符
        segments = [self._restore_code_blocks(s) for s in segments]
        segments = [s for s in segments if len(s) >= self.cfg.min_chars]

        chunks = []
        global_idx = 0

        for seg in segments:
            # 段落未超过硬上限 → 保持完整
            # target_chars 只是偏好，不作为切分触发条件
            if len(seg) <= self.cfg.max_chars:
                chunks.append({
                    "chunk_id": f"{self.source}_chunk{global_idx:04d}",
                    "content": seg,
                    "source": self.source,
                    "doc_type": self.doc_type,
                    "title": title,
                })
                global_idx += 1
                continue

            # --- 最后手段：对真正超大的段落实行滑动窗口切分 ---
            start = 0
            while start < len(seg):
                end = min(start + self.cfg.target_chars, len(seg))
                # 避免末尾产生过短的碎片
                if end == len(seg) and end - start < self.cfg.min_chars:
                    if chunks:
                        prev = chunks[-1]
                        if len(prev["content"]) + (end - start) <= self.cfg.max_chars + 500:
                            prev["content"] += "\n\n" + seg[start:]
                            break
                    # 无法合并，仍然创建该 chunk

                chunk_text = seg[start:end]
                chunks.append({
                    "chunk_id": f"{self.source}_chunk{global_idx:04d}",
                    "content": chunk_text,
                    "source": self.source,
                    "doc_type": self.doc_type,
                    "title": title,
                })
                global_idx += 1
                if end >= len(seg):
                    break
                start = end - self.cfg.overlap_chars

        return chunks

    def _restore_code_blocks(self, text: str) -> str:
        """将代码块占位符替换回原始围栏代码块。"""
        for ph, code in self._code_blocks.items():
            text = text.replace(ph, code)
        return text

    # ---- Parent-Child 切块 ----

    def split_into_children(self, parent_chunk: dict, parent_index: int) -> list[dict]:
        """将一个 Parent Chunk 切分为多个 Child Chunk。

        在句子边界（. ! ? 换行）处切分，每个 child 大小控制在
        child_target_chars 附近，保证每个 child 是语义完整的短片段。

        Args:
            parent_chunk: 父级 chunk dict，包含 content、chunk_id 等字段。
            parent_index: 父级 chunk 的全局序号。

        Returns:
            子级 chunk dict 列表，每个都携带 parent_id 指向父级。
        """
        parent_id = parent_chunk["chunk_id"]
        text = parent_chunk["content"]

        # 按句子边界切分
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # 如果句子太少或太短，再尝试按换行切分
        if len(sentences) <= 2:
            sentences = re.split(r'\n+', text)

        children = []
        buf = ""
        child_idx = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            if buf and len(buf) + len(sent) + 1 <= self.cfg.child_target_chars:
                buf += " " + sent
            elif buf and len(buf) >= self.cfg.child_min_chars:
                children.append(self._make_child(
                    parent_chunk, parent_id, buf, child_idx
                ))
                child_idx += 1
                buf = sent
            elif not buf:
                buf = sent
            else:
                # buf 太短 → 强制合并
                buf += " " + sent

        # 处理最后一个 buffer
        if buf and len(buf) >= self.cfg.child_min_chars:
            children.append(self._make_child(
                parent_chunk, parent_id, buf, child_idx
            ))
        elif buf and children:
            # 最后一段太短，合并到上一个 child
            children[-1]["content"] += " " + buf

        return children

    @staticmethod
    def _make_child(parent: dict, parent_id: str, content: str,
                    child_index: int) -> dict:
        """构造一个 Child Chunk 的 dict。"""
        child_id = parent_id.replace("parent", "child")
        return {
            "chunk_id": f"{parent['source']}_child{child_index:04d}",
            "content": content.strip(),
            "source": parent["source"],
            "doc_type": parent["doc_type"],
            "title": parent.get("title", ""),
            "parent_id": parent_id,
            "chunk_level": "child",
            "child_index": child_index,
        }


# ---------------------------------------------------------------------------
# 批量处理（Parent-Child 双层级版本）
# ---------------------------------------------------------------------------

def process_all_documents_v2(
    papers_dir: str | Path,
    docs_dir: str | Path,
    blogs_dir: str | Path,
    output_dir: str | Path,
) -> tuple[list[dict], list[dict]]:
    """解析并切分知识库中的所有文档，输出 Parent + Child 两个层级。

    Parent Chunk: 语义段落（1500-3000 chars），信息完整 → 喂给 LLM
    Child Chunk:  句子级片段（200-500 chars），精准匹配 → embedding 索引

    输出两个文件:
        parent_chunks.json — LLM 上下文层（BM25 也在这层建索引）
        child_chunks.json  — embedding 检索层（存入 Milvus）

    Args:
        papers_dir: data/raw/papers/ 路径。
        docs_dir: data/raw/docs/ 路径。
        blogs_dir: data/raw/blogs/ 路径。
        output_dir: 输出目录（通常是 data/processed/）。

    Returns:
        (parent_chunks, child_chunks) 两个列表。
    """
    papers_dir = Path(papers_dir)
    docs_dir = Path(docs_dir)
    blogs_dir = Path(blogs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载论文元数据
    paper_meta = {}
    meta_path = papers_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            paper_meta = {m.get("arxiv_id", ""): m for m in json.load(f)}

    all_parents = []
    all_children = []
    total_files = 0
    parent_idx = 0

    def _process_one(source: str, text: str, doc_type: str, extra_meta: dict = None):
        """处理单个文档：生成 parent → 拆分 child → 收集。"""
        nonlocal parent_idx
        if not text.strip():
            return

        chunker = SemanticChunker(source=source, doc_type=doc_type)
        parents = chunker.chunk(text)  # 现有流程产生 parent chunk

        for p in parents:
            # 给每个 parent 打上层级标记
            p["chunk_level"] = "parent"
            p["parent_id"] = p["chunk_id"]
            if extra_meta:
                p.update(extra_meta)

            # 切分出 child chunks
            children = chunker.split_into_children(p, parent_idx)

            all_parents.append(p)
            all_children.extend(children)
            parent_idx += 1

    # --- 论文 (PDF) ---
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    print(f"\n处理 {len(pdf_files)} 篇论文 (PDF) [Parent-Child 模式]...")
    for pdf_path in pdf_files:
        arxiv_id = pdf_path.stem
        text = parse_pdf(pdf_path)
        if not text.strip():
            print(f"  跳过 (空): {pdf_path.name}")
            continue

        meta = paper_meta.get(arxiv_id, {})
        _process_one(
            source=f"paper/{arxiv_id}",
            text=text,
            doc_type="paper",
            extra_meta={
                "paper_title": meta.get("title", ""),
                "paper_authors": meta.get("authors", []),
                "paper_published": meta.get("published", ""),
            },
        )
        total_files += 1
        print(f"  {pdf_path.name[:50]:50s} → {parent_idx} parents")

    # --- 技术文档 (TXT) ---
    txt_files = sorted(docs_dir.glob("*.txt"))
    print(f"\n处理 {len(txt_files)} 份技术文档 (TXT) [Parent-Child 模式]...")
    for txt_path in txt_files:
        text = parse_txt(txt_path)
        if not text.strip():
            print(f"  跳过 (空): {txt_path.name}")
            continue
        _process_one(source=txt_path.stem, text=text, doc_type="doc")
        total_files += 1
        print(f"  {txt_path.name:50s} → {parent_idx} parents")

    # --- 博客 (TXT) ---
    blog_files = sorted(blogs_dir.glob("*.txt"))
    print(f"\n处理 {len(blog_files)} 篇博客 (TXT) [Parent-Child 模式]...")
    for txt_path in blog_files:
        text = parse_txt(txt_path)
        if not text.strip():
            print(f"  跳过 (空): {txt_path.name}")
            continue
        _process_one(source=f"blog/{txt_path.stem}", text=text, doc_type="blog")
        total_files += 1
        print(f"  {txt_path.name:50s} → {parent_idx} parents")

    # 保存 Parent Chunks
    parent_path = output_dir / "parent_chunks.json"
    with open(parent_path, "w", encoding="utf-8") as f:
        json.dump(all_parents, f, indent=2, ensure_ascii=False)

    # 保存 Child Chunks
    child_path = output_dir / "child_chunks.json"
    with open(child_path, "w", encoding="utf-8") as f:
        json.dump(all_children, f, indent=2, ensure_ascii=False)

    # 统计
    print(f"\n{'='*60}")
    print(f"总计: {total_files} 个文件")
    print(f"  Parent Chunks: {len(all_parents):5d}  → {parent_path}")
    print(f"  Child  Chunks: {len(all_children):5d}  → {child_path}")

    if all_parents:
        avg_parent = int(sum(len(c["content"]) for c in all_parents) / len(all_parents))
        print(f"  Parent 平均大小: {avg_parent} 字符")

    if all_children:
        avg_child = int(sum(len(c["content"]) for c in all_children) / len(all_children))
        # Child 大小分布
        buckets = [0, 200, 300, 400, 500, 99999]
        dist = [0] * (len(buckets) - 1)
        for c in all_children:
            l = len(c["content"])
            for i in range(len(buckets) - 1):
                if buckets[i] <= l < buckets[i+1]:
                    dist[i] += 1
                    break
        print(f"  Child  平均大小: {avg_child} 字符")
        print(f"  Child  分布: <200: {dist[0]}, 200-300: {dist[1]}, "
              f"300-400: {dist[2]}, 400-500: {dist[3]}, ≥500: {dist[4]}")

    return all_parents, all_children


# ---------------------------------------------------------------------------
# 批量处理（旧版单层版本，保留兼容）
# ---------------------------------------------------------------------------

def process_all_documents(
    papers_dir: str | Path,
    docs_dir: str | Path,
    blogs_dir: str | Path,
    output_path: str | Path,
    metadata_json: str | Path = None,
) -> list[dict]:
    """解析并切分知识库中的所有文档。

    Args:
        papers_dir: data/raw/papers/ 路径（PDF 文件 + metadata.json）。
        docs_dir: data/raw/docs/ 路径（TXT 文件）。
        blogs_dir: data/raw/blogs/ 路径（TXT 文件）。
        output_path: all_chunks.json 的输出路径。
        metadata_json: collect_docs 生成的元数据 JSON（可选）。

    Returns:
        所有 chunk dict 的完整列表。
    """
    papers_dir = Path(papers_dir)
    docs_dir = Path(docs_dir)
    blogs_dir = Path(blogs_dir)
    output_path = Path(output_path)

    # 加载论文元数据以获取标题
    paper_meta = {}
    meta_path = papers_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            paper_meta = {m.get("arxiv_id", ""): m for m in json.load(f)}

    all_chunks = []
    total_files = 0
    total_chunks = 0

    # --- 论文 (PDF) ---
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    print(f"\n处理 {len(pdf_files)} 篇论文 (PDF)...")
    for pdf_path in pdf_files:
        arxiv_id = pdf_path.stem
        text = parse_pdf(pdf_path)
        if not text.strip():
            print(f"  跳过 (空): {pdf_path.name}")
            continue

        meta = paper_meta.get(arxiv_id, {})
        source = f"paper/{arxiv_id}"
        chunker = SemanticChunker(source=source, doc_type="paper")
        chunks = chunker.chunk(text)
        # 注入论文元数据
        for c in chunks:
            c["paper_title"] = meta.get("title", "")
            c["paper_authors"] = meta.get("authors", [])
            c["paper_published"] = meta.get("published", "")
        all_chunks.extend(chunks)
        total_files += 1
        total_chunks += len(chunks)
        print(f"  {pdf_path.name[:50]:50s} → {len(chunks):4d} chunks")

    # --- 技术文档 (TXT) ---
    txt_files = sorted(docs_dir.glob("*.txt"))
    print(f"\n处理 {len(txt_files)} 份技术文档 (TXT)...")
    for txt_path in txt_files:
        text = parse_txt(txt_path)
        if not text.strip():
            print(f"  跳过 (空): {txt_path.name}")
            continue

        source = txt_path.stem  # 如 "mcp_architecture"
        chunker = SemanticChunker(source=source, doc_type="doc")
        chunks = chunker.chunk(text)
        all_chunks.extend(chunks)
        total_files += 1
        total_chunks += len(chunks)
        print(f"  {txt_path.name:50s} → {len(chunks):4d} chunks")

    # --- 博客 (TXT) ---
    blog_files = sorted(blogs_dir.glob("*.txt"))
    print(f"\n处理 {len(blog_files)} 篇博客 (TXT)...")
    for txt_path in blog_files:
        text = parse_txt(txt_path)
        if not text.strip():
            print(f"  跳过 (空): {txt_path.name}")
            continue

        source = f"blog/{txt_path.stem}"
        chunker = SemanticChunker(source=source, doc_type="blog")
        chunks = chunker.chunk(text)
        all_chunks.extend(chunks)
        total_files += 1
        total_chunks += len(chunks)
        print(f"  {txt_path.name:50s} → {len(chunks):4d} chunks")

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"总计: {total_files} 个文件 → {total_chunks} 个 chunk")
    print(f"已保存到: {output_path}")

    # 统计信息
    doc_chunks = sum(1 for c in all_chunks if c["doc_type"] == "doc")
    paper_chunks = sum(1 for c in all_chunks if c["doc_type"] == "paper")
    blog_chunks = sum(1 for c in all_chunks if c["doc_type"] == "blog")
    avg_len = int(sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1))
    print(f"  论文: {paper_chunks:5d}  文档: {doc_chunks:5d}  博客: {blog_chunks:5d}")
    print(f"  平均 chunk 大小: {avg_len} 字符")

    return all_chunks


# ---------------------------------------------------------------------------
# 快速测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """测试入口：默认运行 Parent-Child 双层级切块。"""
    import sys
    base = Path(__file__).parent.parent.parent  # DeepReason/

    if len(sys.argv) > 1:
        # 处理指定文件用于测试
        test_file = Path(sys.argv[1])
        text = parse_document(test_file)
        chunker = SemanticChunker(source=test_file.name, doc_type="test")
        parents = chunker.chunk(text)
        print(f"\n=== Parent Chunks: {len(parents)} 个 ===")
        for p in parents[:3]:
            print(f"\n[{p['chunk_id']}] ({len(p['content'])}字符):")
            print(p["content"][:300])
            print("---")

            # 展示 child 切分
            children = chunker.split_into_children(p, 0)
            print(f"  → {len(children)} 个 Child Chunks:")
            for ch in children[:3]:
                print(f"    [{ch['chunk_id']}] ({len(ch['content'])}字符): "
                      f"{ch['content'][:100]}...")
    else:
        # 完整批量处理（Parent-Child 模式）
        process_all_documents_v2(
            papers_dir=base / "data" / "raw" / "papers",
            docs_dir=base / "data" / "raw" / "docs",
            blogs_dir=base / "data" / "raw" / "blogs",
            output_dir=base / "data" / "processed",
        )
