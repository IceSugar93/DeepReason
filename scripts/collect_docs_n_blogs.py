"""Collect technical documentation and authoritative blog posts.

Fetches curated HTML pages (framework docs + influential AI blogs), extracts
the main text content, and saves it as plain text files under data/raw/docs/
and data/raw/blogs/.

The URL lists are hand-picked to cover only core concepts relevant to the
DeepReason project (RAG, LLM agents, tool use, multi-agent debate, reflexion,
MCP, LangGraph).
"""

import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Optional HTTP proxy. Set to None to disable.
HTTP_PROXY = "http://127.0.0.1:7890"
# HTTP_PROXY = None

REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 5

# Minimum extracted text length (chars). Pages shorter than this are likely
# extraction failures or redirect/JS-only pages — flag them for review.
MIN_TEXT_LENGTH = 500

# Block-level HTML tags that should be surrounded by newlines during extraction.
# Everything else (code, span, a, strong, em, ...) stays inline so inline code
# and JSON snippets don't get fragmented across many short lines.
BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "blockquote", "pre", "tr", "br",
              "ul", "ol", "table", "thead", "tbody"]


# --------------------------------------------------------------------------
# Curated URL lists
# --------------------------------------------------------------------------

# Core concept pages only — the project's knowledge backbone.
DOC_URLS: list[tuple[str, str]] = [
    # --- MCP (the protocol this project's MCP Server is based on) ---
    ("mcp_introduction", "https://modelcontextprotocol.io/introduction"),
    ("mcp_architecture", "https://modelcontextprotocol.io/docs/concepts/architecture"),
    ("mcp_tools", "https://modelcontextprotocol.io/docs/concepts/tools"),
    ("mcp_resources", "https://modelcontextprotocol.io/docs/concepts/resources"),
    ("mcp_prompts", "https://modelcontextprotocol.io/docs/concepts/prompts"),
    ("mcp_sampling", "https://modelcontextprotocol.io/docs/concepts/sampling"),
    ("mcp_transports", "https://modelcontextprotocol.io/docs/concepts/transports"),
    ("mcp_specification", "https://modelcontextprotocol.io/specification"),

    # --- LangGraph (the state-machine framework used in this project) ---
    # NOTE: LangGraph docs moved from langchain-ai.github.io to docs.langchain.com.
    # The old URLs return a JS "Redirecting..." page. These are the new core pages.
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

    # --- LangChain core concepts (Agent / Chat Model background) ---
    # NOTE: python.langchain.com/docs/concepts/{tools,runnables} now redirect to the
    # agents page (identical content), so they are omitted to avoid duplicates.
    ("langchain_agents", "https://python.langchain.com/docs/concepts/agents/"),
    ("langchain_chat_models", "https://python.langchain.com/docs/concepts/chat_models/"),
]

# Authoritative blog posts from the most influential people in the field.
# URLs verified against each blog's index page.
BLOG_URLS: list[tuple[str, str]] = [
    # --- Lilian Weng (OpenAI former Head of Research) — foundational agent/RAG posts ---
    ("lilianweng_llm_agents", "https://lilianweng.github.io/posts/2023-06-23-agent/"),
    ("lilianweng_prompt_engineering", "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"),
    ("lilianweng_adv_attack_llm", "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"),

    # --- Jay Alammar — the canonical Transformer visual explainer ---
    ("jalammar_illustrated_transformer", "https://jalammar.github.io/illustrated-transformer/"),
    ("jalammar_illustrated_gpt2", "https://jalammar.github.io/illustrated-gpt2/"),

    # --- Eugene Yan — pragmatic, widely-referenced applied-ML writing ---
    ("eugeneyan_llm_patterns", "https://eugeneyan.com/writing/llm-patterns/"),
    ("eugeneyan_qa_evals", "https://eugeneyan.com/writing/qa-evals/"),
    ("eugeneyan_llm_evals", "https://eugeneyan.com/writing/evals/"),
]


# --------------------------------------------------------------------------
# Fetch + extract
# --------------------------------------------------------------------------

def fetch_html(url: str) -> str | None:
    """Fetch HTML content with retries and proxy support."""
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
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


def extract_main_text(html: str) -> str:
    """Extract readable main text from HTML, stripping boilerplate.

    Uses block-aware extraction: inline elements (code, span, a, ...) stay on
    one line so JSON/code samples don't get fragmented, while block elements
    (p, h1-6, li, ...) break cleanly with newlines.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags entirely.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                    "form", "button", "svg", "iframe", "noscript"]):
        tag.decompose()

    # Prefer semantic content containers if present.
    main = (soup.find("main") or soup.find("article")
            or soup.find(id=re.compile(r"content|main|article", re.I))
            or soup.find(class_=re.compile(r"content|markdown|prose|article", re.I))
            or soup.find("body"))
    if main is None:
        main = soup

    # Insert newline markers at block-level boundaries. This keeps inline text
    # (including inline <code>) joined on one line, while paragraphs, headings,
    # and list items each break onto their own line.
    for tag in main.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = main.get_text()  # no separator: inline elements join inline
    # Collapse runs of spaces/tabs within each line, strip and drop empty lines.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def save_text(text: str, save_path: Path) -> None:
    """Save extracted text to a file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(text, encoding="utf-8")


def collect_sources(url_list: list[tuple[str, str]], out_dir: Path,
                    label: str) -> list[dict]:
    """Fetch, extract, and save a list of (name, url) sources.

    Removes orphan .txt files (from old/renamed URLs) before fetching, and
    re-fetches any existing file whose content is too short (likely a failed
    or redirect page).

    Returns metadata records for each source.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_names = {name for name, _ in url_list}

    # Clean up orphaned files: .txt files whose name is no longer in the URL list.
    for old_file in out_dir.glob("*.txt"):
        if old_file.stem not in valid_names:
            print(f"Removing orphan file: {old_file.name}")
            old_file.unlink()

    metadata = []

    for name, url in url_list:
        save_path = out_dir / f"{name}.txt"

        # Skip if already downloaded with substantial content.
        if save_path.exists() and len(save_path.read_text(encoding="utf-8")) > MIN_TEXT_LENGTH:
            print(f"Skipped (exists): {name}")
            metadata.append({"name": name, "url": url, "path": str(save_path), "skipped": True})
            continue

        print(f"Fetching [{label}]: {name}  ->  {url}")
        html = fetch_html(url)
        if html is None:
            print(f"  FAILED to fetch: {name}")
            metadata.append({"name": name, "url": url, "path": str(save_path), "fetch_failed": True})
            continue

        text = extract_main_text(html)
        if len(text) < MIN_TEXT_LENGTH:
            print(f"  WARNING: extracted text too short ({len(text)} chars), may need review: {name}")

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

    docs_metadata = collect_sources(DOC_URLS, RAW_DIR / "docs", label="doc")
    blog_metadata = collect_sources(BLOG_URLS, RAW_DIR / "blogs", label="blog")

    all_metadata = {"docs": docs_metadata, "blogs": blog_metadata}
    meta_path = RAW_DIR / "docs_blogs_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    # Stats
    docs_ok = sum(1 for m in docs_metadata if not m.get("fetch_failed"))
    blog_ok = sum(1 for m in blog_metadata if not m.get("fetch_failed"))
    print(f"\n{'='*60}")
    print(f"Docs:   {docs_ok}/{len(DOC_URLS)} fetched successfully")
    print(f"Blogs:  {blog_ok}/{len(BLOG_URLS)} fetched successfully")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
