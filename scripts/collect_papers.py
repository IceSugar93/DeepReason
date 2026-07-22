"""数据采集脚本 — 下载 arXiv 论文

围绕 DeepReason 项目主题下载相关论文:
RAG、LLM Agent、工具使用/Function Calling、多Agent辩论、
Reflexion/自纠错、MCP 协议。

兼容 arxiv 库 v4.x（PDF 通过 httpx 下载，因为 v4 的 Result 对象
不再提供 download_pdf() 方法）。
"""

import json
import time
from pathlib import Path

import arxiv
import httpx

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# 可选 HTTP 代理（设为 None 则禁用）。如果本地有代理（如 Clash 在 7890 端口）
# 且 arXiv 访问慢/被墙，保持下面这行。
HTTP_PROXY = "http://127.0.0.1:7890"
# HTTP_PROXY = None

# 请求设置
REQUEST_TIMEOUT = 60      # 单次 PDF 下载超时（秒）
MAX_RETRIES = 3           # 下载失败重试次数
RETRY_DELAY = 5           # 重试间隔（秒）


def download_pdf(pdf_url: str, save_path: Path) -> bool:
    """通过 httpx 下载单篇 PDF，带重试和代理支持。

    Args:
        pdf_url: PDF 下载地址。
        save_path: 本地保存路径。

    Returns:
        下载成功返回 True，失败返回 False。
    """
    # httpx >= 0.28 使用 `proxy`（单数）；旧版本使用 `proxies`
    client_kwargs = {"timeout": REQUEST_TIMEOUT, "follow_redirects": True}
    if HTTP_PROXY:
        client_kwargs["proxy"] = HTTP_PROXY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(**client_kwargs) as client:
                response = client.get(pdf_url)
                response.raise_for_status()

                # 验证响应确实是 PDF（arXiv 有时返回 HTML 错误页）
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type and not response.content[:5] == b"%PDF-":
                    raise ValueError(f"非预期的 content-type: {content_type}")

                save_path.write_bytes(response.content)
                return True
        except Exception as e:
            print(f"  第 {attempt}/{MAX_RETRIES} 次尝试失败: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return False


def download_arxiv_papers(query: str, max_results: int = 20) -> list[dict]:
    """按关键词搜索并下载 arXiv 论文。

    Args:
        query: arXiv 搜索查询（关键词，如 "retrieval augmented generation"）。
        max_results: 此查询最多下载的论文数。

    Returns:
        已下载论文的元数据 dict 列表。
    """
    paper_dir = RAW_DIR / "papers"
    paper_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    # 先物化完整结果列表。arXiv 的 results() 生成器是延迟分页的，
    # 如果在迭代生成器的同时进行慢速网络下载，可能导致生成器提前终止。
    # list() 强制一次性拉取所有分页结果。
    results = list(client.results(search))

    metadata = []
    for result in results:
        arxiv_id = result.get_short_id()
        title = result.title
        pdf_path = paper_dir / f"{arxiv_id}.pdf"
        pdf_url = result.pdf_url

        # 无论下载成功与否都构建元数据记录
        record = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "categories": result.categories,
            "published": str(result.published),
            "pdf_url": pdf_url,
            "pdf_path": str(pdf_path),
            "query": query,
        }

        # 已存在且非空 → 跳过
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"跳过 (已存在): {title[:60]}")
            metadata.append(record)
            continue

        # 通过 httpx 下载 PDF
        print(f"下载中: {title[:60]}")
        success = download_pdf(pdf_url, pdf_path)
        if success:
            metadata.append(record)
        else:
            print(f"  下载失败: {arxiv_id}")
            record["download_failed"] = True
            metadata.append(record)

    return metadata


def deduplicate(metadata: list[dict]) -> list[dict]:
    """按 arxiv_id 去重，保留首次出现的那条。"""
    seen = set()
    unique = []
    for m in metadata:
        if m["arxiv_id"] not in seen:
            seen.add(m["arxiv_id"])
            unique.append(m)
    return unique


if __name__ == "__main__":
    # 与 DeepReason 项目技术方向对齐的关键词查询
    # 每个查询拉取与该技术支柱最相关的论文
    queries = [
        "retrieval augmented generation RAG",
        "LLM agent tool use function calling",
        "multi-agent debate reasoning",
        "reflexion self-correction LLM",
        "MCP model context protocol agent",
    ]

    all_metadata = []
    for q in queries:
        print(f"\n=== 搜索: {q} ===")
        meta = download_arxiv_papers(q, max_results=20)
        all_metadata.extend(meta)

    # 跨查询去重（一篇论文可能匹配多个关键词）
    unique_metadata = deduplicate(all_metadata)

    # 保存合并后的元数据
    paper_dir = RAW_DIR / "papers"
    with open(paper_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(unique_metadata, f, indent=2, ensure_ascii=False)

    # 统计
    downloaded = sum(1 for m in unique_metadata if not m.get("download_failed"))
    failed = sum(1 for m in unique_metadata if m.get("download_failed"))
    print(f"\n{'='*60}")
    print(f"去重后论文总计: {len(unique_metadata)}")
    print(f"  成功下载: {downloaded}")
    print(f"  下载失败: {failed}")
    print(f"元数据已保存到: {paper_dir / 'metadata.json'}")
