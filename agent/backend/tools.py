from typing import Any, Callable, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool

from constants import (
    DEFAULT_SEARCH_RESULTS,
    MAX_SEARCH_RESULTS,
    SEARCH_TOOL_TAVILY,
    SEARCH_TOOL_SERP,
)
from logger import logger


def make_tavily_tool(
    tavily_api_key: Optional[str],
    max_results: int = DEFAULT_SEARCH_RESULTS
) -> TavilySearchResults:
    return TavilySearchResults(
        max_results=max_results,
        tavily_api_key=tavily_api_key,
        include_answer=True,
        include_raw_content=True
    )


def make_serp_tool(
    serp_api_key: Optional[str],
    max_results: int = DEFAULT_SEARCH_RESULTS
) -> Tool:
    search_wrapper = SerpAPIWrapper(serpapi_api_key=serp_api_key)

    def serp_search(query: str) -> list[dict[str, Any]]:
        try:
            results = search_wrapper.results(query)

            formatted_results: list[dict[str, Any]] = []
            organic_results = results.get("organic_results", [])[:max_results]

            for item in organic_results:
                formatted_results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "content": item.get("snippet", ""),
                    "source": "serp"
                })

            return formatted_results

        except Exception as e:
            logger.error(SEARCH_TOOL_SERP, f"Search failed for query '{query}': {e}")
            return []

    return Tool(
        name="serp_search",
        description="Search using SerpAPI for current web results",
        func=serp_search
    )


def dedupe_keep_best(
    items: list[dict[str, Any]],
    max_items: int = MAX_SEARCH_RESULTS
) -> list[dict[str, Any]]:
    seen_urls: set[str] = set()
    unique_items: list[dict[str, Any]] = []

    for item in items:
        url = item.get("url") or item.get("source")

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        unique_items.append(item)

    return unique_items[:max_items]


def merge_and_rank_results(
    tavily_results: list[dict[str, Any]],
    serp_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    max_length = max(len(tavily_results), len(serp_results))

    for i in range(max_length):
        if i < len(tavily_results):
            result = tavily_results[i].copy()
            result["source"] = SEARCH_TOOL_TAVILY
            merged.append(result)

        if i < len(serp_results):
            result = serp_results[i].copy()
            result["source"] = SEARCH_TOOL_SERP
            merged.append(result)

    return dedupe_keep_best(merged)


def execute_search_queries(
    tool: Any,
    queries: list[str],
    tool_name: str
) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []

    for idx, query in enumerate(queries, 1):
        logger.info(tool_name, f"Query {idx}/{len(queries)}: {query}", force_flush=True)

        try:
            if hasattr(tool, 'run'):
                results = tool.run(query)
            else:
                results = tool.func(query)

            if isinstance(results, list):
                for result in results:
                    result["query"] = query
                    result["search_tool"] = tool_name

                all_results.extend(results)
                logger.info(
                    tool_name,
                    f"Query {idx} returned {len(results)} results",
                    force_flush=True
                )

        except Exception as e:
            logger.error(tool_name, f"Query {idx} failed: {e}", force_flush=True)

    return all_results