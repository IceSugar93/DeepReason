"""采集技术文档和权威博客文章

抓取精选的 HTML 页面（框架文档 + 有影响力的 AI 博客），提取正文内容，
保存为纯文本文件到 data/raw/docs/ 和 data/raw/blogs/。

URL 列表是手工精选的，只覆盖与 DeepReason 项目直接相关的核心概念
（RAG、LLM Agent、工具使用、多Agent辩论、Reflexion、MCP、LangGraph）。
"""

import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# 可选 HTTP 代理。设为 None 则禁用。
HTTP_PROXY = "http://127.0.0.1:7890"
# HTTP_PROXY = None

REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 5

# 最小提取文本长度（字符）。短于此值的页面很可能是提取失败
# 或重定向/纯 JS 页面 — 标记为需要检查。
MIN_TEXT_LENGTH = 500

# 块级 HTML 标签，提取时在其前后插入换行。
# 其余标签（code, span, a, strong, em, ...）保持行内处理，
# 这样内联代码和 JSON 片段就不会被切碎成多行。
BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "blockquote", "pre", "tr", "br",
              "ul", "ol", "table", "thead", "tbody"]


# --------------------------------------------------------------------------
# 精选 URL 列表
# --------------------------------------------------------------------------

# 只收录核心概念页面 — 项目的知识骨架。
DOC_URLS: list[tuple[str, str]] = [
    # --- MCP（本项目 MCP Server 所基于的协议） ---
    ("mcp_introduction", "https://modelcontextprotocol.io/introduction"),
    ("mcp_architecture", "https://modelcontextprotocol.io/docs/concepts/architecture"),
    ("mcp_tools", "https://modelcontextprotocol.io/docs/concepts/tools"),
    ("mcp_resources", "https://modelcontextprotocol.io/docs/concepts/resources"),
    ("mcp_prompts", "https://modelcontextprotocol.io/docs/concepts/prompts"),
    ("mcp_sampling", "https://modelcontextprotocol.io/docs/concepts/sampling"),
    ("mcp_transports", "https://modelcontextprotocol.io/docs/concepts/transports"),
    ("mcp_specification", "https://modelcontextprotocol.io/specification"),

    # --- LangGraph（本项目使用的状态机框架） ---
    # 注意：LangGraph 文档已从 langchain-ai.github.io 迁移到 docs.langchain.com。
    # 旧 URL 只返回 JS "Redirecting..." 页面。以下是新的核心页面。
    ("langgraph_overview", "https://docs.langchain.com/oss/python/langgraph/overview"),
    ("langgraph_thinking", "https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph"),
    ("langgraph_graph_api", "https://docs.langchain.com/oss/python/langgraph/graph-api"),
    ("langgraph_functional_api", "https://docs.langchain.com/oss/python/langgraph/functional-api"),
    ("langgraph_workflows_agents", "https://docs.langchain.com/oss/python/langgraph/workflows-agents"),
    ("langgraph_agentic_rag", "https://docs.langchain.com/oss/python/langgraph/agentic-rag"),
    ("langgraph_persistence", "https://docs.langchain.com/oss/python/langgraph/persistence"),
    ("langgraph_checkpointers", "https://docs.langchain.com/oss/python/langgraph/checkpointers"),
    ("langgraph_interrupts", "https://docs.langchain.com/oss/python/langgraph/interrupts"),
    ("langgraph_choosing_apis", "https://docs.langchain.com/oss/python/langgraph/choosing-apis"),

    # --- LangChain 核心概念（Agent/Chat Model 背景知识） ---
    # 注意：python.langchain.com/docs/concepts/{tools,runnables} 现在重定向到
    # agents 页面（内容相同），因此省略以避免重复。
    ("langchain_agents", "https://python.langchain.com/docs/concepts/agents/"),
    ("langchain_chat_models", "https://python.langchain.com/docs/concepts/chat_models/"),
]

# 本领域最有影响力人物的权威博客文章。
# URL 均已对各博客首页核实过。
BLOG_URLS: list[tuple[str, str]] = [
    # --- Lilian Weng（OpenAI 前研究主管）— Agent/RAG 领域的奠基性文章 ---
    ("lilianweng_llm_agents", "https://lilianweng.github.io/posts/2023-06-23-agent/"),
    ("lilianweng_prompt_engineering", "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"),
    ("lilianweng_adv_attack_llm", "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"),

    # --- Jay Alammar — Transformer 可视化解释的经典作者 ---
    ("jalammar_illustrated_transformer", "https://jalammar.github.io/illustrated-transformer/"),
    ("jalammar_illustrated_gpt2", "https://jalammar.github.io/illustrated-gpt2/"),

    # --- Eugene Yan — 务实、广泛被引用的应用 ML 写作者 ---
    ("eugeneyan_llm_patterns", "https://eugeneyan.com/writing/llm-patterns/"),
    ("eugeneyan_qa_evals", "https://eugeneyan.com/writing/qa-evals/"),
    ("eugeneyan_llm_evals", "https://eugeneyan.com/writing/evals/"),
]


# --------------------------------------------------------------------------
# 抓取 + 提取
# --------------------------------------------------------------------------

def fetch_html(url: str) -> str | None:
    """带重试和代理支持的 HTML 抓取。"""
    client_kwargs = {"timeout": REQUEST_TIMEOUT, "follow_redirects": True}
    if HTTP_PROXY:
        client_kwargs["proxy"] = HTTP_PROXY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(**client_kwargs) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.text
        except Exception as e:
            print(f"  第 {attempt}/{MAX_RETRIES} 次尝试失败: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


def extract_main_text(html: str) -> str:
    """从 HTML 中提取可读正文，去除样板内容。

    使用块感知提取：行内元素（code, span, a, ...）保持在同行，
    避免 JSON/代码样本被切碎；块级元素（p, h1-h6, li, ...）用换行分隔。
    """
    soup = BeautifulSoup(html, "html.parser")

    # 彻底移除非内容标签。
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                    "form", "button", "svg", "iframe", "noscript"]):
        tag.decompose()

    # 优先使用语义化的内容容器。
    main = (soup.find("main") or soup.find("article")
            or soup.find(id=re.compile(r"content|main|article", re.I))
            or soup.find(class_=re.compile(r"content|markdown|prose|article", re.I))
            or soup.find("body"))
    if main is None:
        main = soup

    # 在块级元素边界处插入换行标记。这样内联文本（包括内联 <code>）
    # 保持在一行内，而段落、标题、列表项各自换行。
    for tag in main.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = main.get_text()  # 不使用分隔符：行内元素自然拼接
    # 压缩行内空格/制表符、去除首尾空白、删除空行
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def save_text(text: str, save_path: Path) -> None:
    """将提取的文本保存到文件。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(text, encoding="utf-8")


def collect_sources(url_list: list[tuple[str, str]], out_dir: Path,
                    label: str) -> list[dict]:
    """抓取、提取并保存一批 (名称, URL) 来源。

    在抓取前清理孤儿 .txt 文件（来自旧/重命名 URL），
    对内容过短的已有文件（可能是失败或重定向页面）重新抓取。

    Returns:
        每个来源的元数据记录列表。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_names = {name for name, _ in url_list}

    # 清理孤儿文件：名称不在当前 URL 列表中的 .txt 文件
    for old_file in out_dir.glob("*.txt"):
        if old_file.stem not in valid_names:
            print(f"删除孤儿文件: {old_file.name}")
            old_file.unlink()

    metadata = []

    for name, url in url_list:
        save_path = out_dir / f"{name}.txt"

        # 已有足够内容 → 跳过
        if save_path.exists() and len(save_path.read_text(encoding="utf-8")) > MIN_TEXT_LENGTH:
            print(f"跳过 (已存在): {name}")
            metadata.append({"name": name, "url": url, "path": str(save_path), "skipped": True})
            continue

        print(f"抓取 [{label}]: {name}  ->  {url}")
        html = fetch_html(url)
        if html is None:
            print(f"  抓取失败: {name}")
            metadata.append({"name": name, "url": url, "path": str(save_path), "fetch_failed": True})
            continue

        text = extract_main_text(html)
        if len(text) < MIN_TEXT_LENGTH:
            print(f"  警告: 提取文本太短 ({len(text)} 字符)，可能需要检查: {name}")

        save_text(text, save_path)
        metadata.append({
            "name": name,
            "url": url,
            "path": str(save_path),
            "char_count": len(text),
        })

    return metadata


def main():
    import json

    docs_metadata = collect_sources(DOC_URLS, RAW_DIR / "docs", label="文档")
    blog_metadata = collect_sources(BLOG_URLS, RAW_DIR / "blogs", label="博客")

    all_metadata = {"docs": docs_metadata, "blogs": blog_metadata}
    meta_path = RAW_DIR / "docs_blogs_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    # 统计
    docs_ok = sum(1 for m in docs_metadata if not m.get("fetch_failed"))
    blog_ok = sum(1 for m in blog_metadata if not m.get("fetch_failed"))
    print(f"\n{'='*60}")
    print(f"文档: {docs_ok}/{len(DOC_URLS)} 抓取成功")
    print(f"博客: {blog_ok}/{len(BLOG_URLS)} 抓取成功")
    print(f"元数据已保存到: {meta_path}")


if __name__ == "__main__":
    main()
