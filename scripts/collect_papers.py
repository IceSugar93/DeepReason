"""Data collection script: download papers, docs, and blogs.

Downloads arXiv papers relevant to the DeepReason project themes:
RAG, LLM agents, tool use / function calling, multi-agent debate,
reflexion / self-correction, and MCP.

Compatible with arxiv library v4.x (uses httpx to download PDFs, since the
Result object no longer exposes a download_pdf() method).
"""

import json
import time
from pathlib import Path

import arxiv
import httpx

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Optional HTTP proxy (set to None to disable). If you have a local proxy
# (e.g. Clash on 7890) and arXiv is slow/blocked, keep the line below.
HTTP_PROXY = "http://127.0.0.1:7890"
# HTTP_PROXY = None

# Request settings
REQUEST_TIMEOUT = 60      # seconds per PDF download
MAX_RETRIES = 3           # retry count on download failure
RETRY_DELAY = 5           # seconds between retries


def download_pdf(pdf_url: str, save_path: Path) -> bool:
    """Download a single PDF via httpx, with retries and proxy support.

    Args:
        pdf_url: URL of the PDF to download.
        save_path: Local path to save the PDF.

    Returns:
        True if download succeeded, False otherwise.
    """
    # httpx >= 0.28 uses `proxy` (singular); older versions use `proxies`.
    client_kwargs = {"timeout": REQUEST_TIMEOUT, "follow_redirects": True}
    if HTTP_PROXY:
        client_kwargs["proxy"] = HTTP_PROXY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(**client_kwargs) as client:
                response = client.get(pdf_url)
                response.raise_for_status()

                # Verify the response is actually a PDF (arXiv sometimes returns HTML errors)
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type and not response.content[:5] == b"%PDF-":
                    raise ValueError(f"Unexpected content-type: {content_type}")

                save_path.write_bytes(response.content)
                return True
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return False


def download_arxiv_papers(query: str, max_results: int = 20) -> list[dict]:
    """Download papers from arXiv matching a keyword query.

    Args:
        query: arXiv search query (keywords, e.g. "retrieval augmented generation").
        max_results: Maximum number of papers to download for this query.

    Returns:
        List of metadata dicts for the downloaded papers.
    """
    paper_dir = RAW_DIR / "papers"
    paper_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    # Materialize the full result list first. arXiv's results() generator paginates
    # lazily, and interleaving slow network downloads with generator iteration can
    # cause it to stop early. list() forces all pages to be fetched up front.
    results = list(client.results(search))

    metadata = []
    for result in results:
        arxiv_id = result.get_short_id()
        title = result.title
        pdf_path = paper_dir / f"{arxiv_id}.pdf"
        pdf_url = result.pdf_url

        # Build metadata record regardless of download success
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

        # Skip if already downloaded
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"Skipped (exists): {title[:60]}")
            metadata.append(record)
            continue

        # Download PDF via httpx
        print(f"Downloading: {title[:60]}")
        success = download_pdf(pdf_url, pdf_path)
        if success:
            metadata.append(record)
        else:
            print(f"  FAILED to download: {arxiv_id}")
            record["download_failed"] = True
            metadata.append(record)

    return metadata


def deduplicate(metadata: list[dict]) -> list[dict]:
    """Remove duplicate papers by arxiv_id, keeping first occurrence."""
    seen = set()
    unique = []
    for m in metadata:
        if m["arxiv_id"] not in seen:
            seen.add(m["arxiv_id"])
            unique.append(m)
    return unique


if __name__ == "__main__":
    # Keyword queries aligned with DeepReason project themes.
    # Each query pulls papers most relevant to a specific technical pillar.
    queries = [
        "retrieval augmented generation RAG",
        "LLM agent tool use function calling",
        "multi-agent debate reasoning",
        "reflexion self-correction LLM",
        "MCP model context protocol agent",
    ]

    all_metadata = []
    for q in queries:
        print(f"\n=== Searching: {q} ===")
        meta = download_arxiv_papers(q, max_results=20)
        all_metadata.extend(meta)

    # Deduplicate across queries (a paper may match multiple keywords)
    unique_metadata = deduplicate(all_metadata)

    # Save combined metadata
    paper_dir = RAW_DIR / "papers"
    with open(paper_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(unique_metadata, f, indent=2, ensure_ascii=False)

    # Stats
    downloaded = sum(1 for m in unique_metadata if not m.get("download_failed"))
    failed = sum(1 for m in unique_metadata if m.get("download_failed"))
    print(f"\n{'='*60}")
    print(f"Total unique papers: {len(unique_metadata)}")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed to download: {failed}")
    print(f"Metadata saved to: {paper_dir / 'metadata.json'}")
