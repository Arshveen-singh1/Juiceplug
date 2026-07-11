"""
Built-in ``web_search`` tool — DuckDuckGo backend (zero config, no API key).

If the environment variable ``JUICEPLUG_SEARCH_BACKEND`` is set to
``tavily``, ``serper``, or ``bing``, the corresponding paid API is used
instead (requires the matching API-key env-var).  DuckDuckGo remains the
zero-config fallback.
"""

from __future__ import annotations

import logging
import os
from typing import List

from juiceplug.tools import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

_BACKEND = os.environ.get("JUICEPLUG_SEARCH_BACKEND", "duckduckgo").lower()


def _ddg_search(query: str, max_results: int = 5) -> List[dict]:
    """Search with DuckDuckGo (default, no API key)."""
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise ImportError(
            "duckduckgo-search is required for the default web_search tool. "
            "Install it: pip install duckduckgo-search>=6.1"
        ) from exc

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """Search with Tavily (requires TAVILY_API_KEY)."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — falling back to DuckDuckGo.")
        return _ddg_search(query, max_results)
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "tavily-python is required when JUICEPLUG_SEARCH_BACKEND=tavily. "
            "Install it: pip install tavily-python"
        ) from exc
    client = TavilyClient(api_key=api_key)
    resp = client.search(query, max_results=max_results)
    return resp.get("results", [])


def _serper_search(query: str, max_results: int = 5) -> List[dict]:
    """Search with Serper.dev (requires SERPER_API_KEY)."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        logger.warning("SERPER_API_KEY not set — falling back to DuckDuckGo.")
        return _ddg_search(query, max_results)
    import json
    import urllib.request

    payload = json.dumps({"q": query, "num": max_results}).encode()
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return data.get("organic", [])


def _bing_search(query: str, max_results: int = 5) -> List[dict]:
    """Search with Bing Web Search API (requires BING_API_KEY)."""
    api_key = os.environ.get("BING_API_KEY")
    if not api_key:
        logger.warning("BING_API_KEY not set — falling back to DuckDuckGo.")
        return _ddg_search(query, max_results)
    import json
    import urllib.request
    import urllib.parse

    url = (
        "https://api.bing.microsoft.com/v7.0/search?"
        + urllib.parse.urlencode({"q": query, "count": max_results})
    )
    req = urllib.request.Request(url, headers={"Ocp-Apim-Subscription-Key": api_key})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return data.get("webPages", {}).get("value", [])


_SEARCH_FN = {
    "duckduckgo": _ddg_search,
    "tavily": _tavily_search,
    "serper": _serper_search,
    "bing": _bing_search,
}


# ---------------------------------------------------------------------------
# Registered tool
# ---------------------------------------------------------------------------


@register_tool("web_search")
def web_search(query: str) -> str:
    """Search the web for *query* and return a condensed text summary.

    The backend is chosen via the ``JUICEPLUG_SEARCH_BACKEND`` env-var
    (``duckduckgo`` | ``tavily`` | ``serper`` | ``bing``).  Defaults to
    DuckDuckGo (no API key required).

    Returns
    -------
    str
        A newline-separated summary of results suitable for injection into a
        model prompt as an ``<observation>``.
    """
    search_fn = _SEARCH_FN.get(_BACKEND, _ddg_search)
    logger.info("web_search | backend=%s | query=%r", _BACKEND, query)

    try:
        results = search_fn(query, max_results=5)
    except Exception as exc:
        logger.error("web_search failed: %s", exc)
        return f"[web_search error: {exc}]"

    if not results:
        return "[No results found.]"

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body") or r.get("snippet") or r.get("content") or ""
        href = r.get("href") or r.get("url") or r.get("link") or ""
        lines.append(f"{i}. {title}\n   {body}\n   {href}")

    summary = "\n".join(lines)
    logger.info("web_search returned %d chars", len(summary))
    return summary
