import sys
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from constants import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TEMPERATURE,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    DETAILED_QUERY_COUNT_OPENAI,
    DETAILED_QUERY_COUNT_GEMINI,
    SIMPLE_QUERY_COUNT_OPENAI,
    SIMPLE_QUERY_COUNT_GEMINI,
    TEMPLATE_DETAILED_REPORT,
    TEMPLATE_TWO_COLUMN,
    SEARCH_TOOL_TAVILY,
    SEARCH_TOOL_SERP,
    GEMINI_MAX_SOURCES_LIMIT,
)
from templates import REPORT_TEMPLATES, add_provider_specific_instructions
from tools import (
    make_tavily_tool,
    make_serp_tool,
    merge_and_rank_results,
    dedupe_keep_best,
    execute_search_queries,
)
from settings import settings
from logger import logger, log
from exceptions import LLMTimeoutError


def initial_state(
    query: str,
    config: dict[str, Any],
    messages: list[dict[str, Any]]
) -> dict[str, Any]:

    return {
        "query": query,
        "config": config,
        "messages": messages,
        "plan": "",
        "search_results": [],
        "sources": [],
        "report": None,
        "tavily_report": None,
        "serp_report": None,
    }


def get_llm(provider: str, model: Optional[str]):
    log(f"[LLM] Requested provider={provider}, model={model}")

    actual_provider = settings.get_available_provider(provider)

    if actual_provider != provider:
        log(f"[LLM] ⚠️  Provider changed: {provider} → {actual_provider}")

    if actual_provider == PROVIDER_GEMINI:
        return ChatGoogleGenerativeAI(
            model=model or DEFAULT_GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=DEFAULT_TEMPERATURE,
            max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
            max_retries=settings.GEMINI_MAX_RETRIES,
            timeout=settings.GEMINI_TIMEOUT_SECONDS,
            request_timeout=settings.GEMINI_REQUEST_TIMEOUT,
        )

    return ChatOpenAI(
        model=model or settings.MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=DEFAULT_TEMPERATURE
    )


def invoke_llm_safe(
    llm,
    messages: list[dict[str, str]],
    is_gemini: bool = False,
    timeout_seconds: Optional[int] = None
):
    if not is_gemini:
        return llm.invoke(messages)

    if timeout_seconds is None:
        timeout_seconds = settings.GEMINI_TIMEOUT_SECONDS

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm.invoke, messages)
            try:
                result = future.result(timeout=timeout_seconds)
                sys.stdout.flush()
                return result
            except FuturesTimeoutError:
                error_msg = f"Gemini request exceeded {timeout_seconds}s timeout"
                log(f"[LLM] {error_msg}", force_flush=True)
                raise LLMTimeoutError(error_msg)
    except LLMTimeoutError:
        raise
    except Exception as e:
        log(f"[LLM] Error during Gemini invocation: {e}", force_flush=True)
        raise


def step_plan(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("Graph", "PLAN - Starting planning phase...", force_flush=True)

    # Get actual provider (with fallback if needed)
    requested_provider = state["config"]["provider"]
    actual_provider = settings.get_available_provider(requested_provider)

    # Update config if provider changed
    if actual_provider != requested_provider:
        state["config"]["provider"] = actual_provider
        logger.info(
            "Graph",
            f"PLAN - Provider updated: {requested_provider} → {actual_provider}",
            force_flush=True
        )

    llm = get_llm(actual_provider, state["config"].get("model"))

    is_detailed = state["config"]["template"] == TEMPLATE_DETAILED_REPORT
    is_gemini = actual_provider == PROVIDER_GEMINI

    if is_gemini:
        query_count = DETAILED_QUERY_COUNT_GEMINI if is_detailed else SIMPLE_QUERY_COUNT_GEMINI
    else:
        query_count = DETAILED_QUERY_COUNT_OPENAI if is_detailed else SIMPLE_QUERY_COUNT_OPENAI

    system_prompt = (
        f"You are a research planner. Break the user query into {query_count} "
        f"specific web searches. Return numbered queries only. Be concise."
    )

    logger.info("Graph", "PLAN - Invoking LLM...", force_flush=True)

    try:
        response = invoke_llm_safe(
            llm,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["query"]}
            ],
            is_gemini=is_gemini
        )
        state["plan"] = response.content
        logger.info(
            "Graph",
            f"PLAN - Complete. Generated {len(response.content)} chars",
            force_flush=True
        )
    except LLMTimeoutError:
        # Fallback to simple plan if timeout occurs
        logger.warning("Graph", "PLAN - Timeout occurred, using fallback plan", force_flush=True)
        state["plan"] = (
            f"1. {state['query']}\n"
            f"2. {state['query']} overview\n"
            f"3. {state['query']} details"
        )

    return state



def step_search(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("Graph", "SEARCH - Starting search phase...", force_flush=True)

    settings.validate_search_requirements()

    actual_provider = state["config"]["provider"]
    is_gemini = actual_provider == PROVIDER_GEMINI

    max_budget = settings.GEMINI_MAX_SEARCHES if is_gemini else settings.MAX_SEARCHES
    budget = min(int(state["config"]["search_budget"]), max_budget)

    logger.info(
        "Graph",
        f"SEARCH - Using budget of {budget} queries (provider: {actual_provider})",
        force_flush=True
    )

    raw_lines = [line for line in state["plan"].split("\n") if line.strip()]
    queries = [q.strip(" -0123456789.\"") for q in raw_lines][:budget]

    use_dual = settings.can_use_dual_search

    if use_dual:
        all_hits = _execute_dual_search(queries)
    else:
        all_hits = _execute_single_search(queries)

    state["search_results"] = all_hits
    state["sources"] = _format_sources(all_hits)

    logger.info("Graph", "SEARCH - Complete", force_flush=True)
    return state


def _execute_dual_search(queries: list[str]) -> list[dict[str, Any]]:
    logger.info(
        "Graph",
        f"SEARCH - Running DUAL search with {len(queries)} queries...",
        force_flush=True
    )

    tavily_tool = make_tavily_tool(settings.TAVILY_API_KEY, max_results=5)
    serp_tool = make_serp_tool(settings.SERP_API_KEY, max_results=5)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_tavily = executor.submit(
            execute_search_queries, tavily_tool, queries, SEARCH_TOOL_TAVILY
        )
        future_serp = executor.submit(
            execute_search_queries, serp_tool, queries, SEARCH_TOOL_SERP
        )

        tavily_hits = future_tavily.result()
        serp_hits = future_serp.result()

    logger.info("Graph", f"SEARCH - Tavily returned {len(tavily_hits)} results", force_flush=True)
    logger.info("Graph", f"SEARCH - SerpAPI returned {len(serp_hits)} results", force_flush=True)

    merged_results = merge_and_rank_results(tavily_hits, serp_hits)
    logger.info(
        "Graph",
        f"SEARCH - Merged to {len(merged_results)} unique sources",
        force_flush=True
    )

    return merged_results


def _execute_single_search(queries: list[str]) -> list[dict[str, Any]]:
    logger.info(
        "Graph",
        f"SEARCH - Running SINGLE search (Tavily) with {len(queries)} queries...",
        force_flush=True
    )

    tavily_tool = make_tavily_tool(settings.TAVILY_API_KEY, max_results=5)
    all_hits = execute_search_queries(tavily_tool, queries, SEARCH_TOOL_TAVILY)

    deduped_results = dedupe_keep_best(all_hits)
    logger.info(
        "Graph",
        f"SEARCH - Deduped to {len(deduped_results)} unique sources",
        force_flush=True
    )

    return deduped_results


def _format_sources(search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for i, hit in enumerate(search_results):
        snippet = hit.get("content") or ""
        sources.append({
            "id": i + 1,
            "title": hit.get("title"),
            "url": hit.get("url"),
            "snippet": snippet[:300],  # Limit snippet length
            "query": hit.get("query"),
            "source": hit.get("search_tool", "unknown")
        })
    return sources



def step_synthesize(state: dict[str, Any]) -> dict[str, Any]:
    logger.info("Graph", "SYNTHESIZE - Starting synthesis phase...", force_flush=True)

    actual_provider = state["config"]["provider"]
    llm = get_llm(actual_provider, state["config"].get("model"))
    template_name = state["config"]["template"]
    query = state["query"]
    sources = state["sources"]
    is_gemini = actual_provider == PROVIDER_GEMINI

    has_tavily = any(s.get("source") == SEARCH_TOOL_TAVILY for s in sources)
    has_serp = any(s.get("source") == SEARCH_TOOL_SERP for s in sources)
    use_dual = settings.can_use_dual_search and has_tavily and has_serp

    if use_dual:
        _execute_dual_synthesis(state, llm, query, sources, template_name, is_gemini)
    else:
        _execute_single_synthesis(state, llm, query, sources, template_name, is_gemini)

    logger.info("Graph", "SYNTHESIZE - Complete", force_flush=True)
    return state


def _execute_dual_synthesis(
    state: dict[str, Any],
    llm,
    query: str,
    sources: list[dict[str, Any]],
    template_name: str,
    is_gemini: bool
) -> None:
    logger.info(
        "Graph",
        "SYNTHESIZE - Generating reports from BOTH search sources...",
        force_flush=True
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_tavily = executor.submit(
            _synthesize_single_report,
            llm, query, sources, template_name, SEARCH_TOOL_TAVILY, is_gemini
        )
        future_serp = executor.submit(
            _synthesize_single_report,
            llm, query, sources, template_name, SEARCH_TOOL_SERP, is_gemini
        )

        try:
            timeout = settings.GEMINI_REQUEST_TIMEOUT if is_gemini else None
            tavily_report = future_tavily.result(timeout=timeout)
            serp_report = future_serp.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error(
                "Graph",
                "SYNTHESIZE - One or both reports timed out",
                force_flush=True
            )
            tavily_report = "Report timed out"
            serp_report = "Report timed out"

    logger.info(
        "Graph",
        f"SYNTHESIZE - Tavily report: {len(tavily_report) if tavily_report else 0} chars",
        force_flush=True
    )
    logger.info(
        "Graph",
        f"SYNTHESIZE - SerpAPI report: {len(serp_report) if serp_report else 0} chars",
        force_flush=True
    )

    winning_tool, final_report = _select_best_report(
        llm, query, tavily_report, serp_report, is_gemini
    )

    filtered_sources = [s for s in sources if s.get("source") == winning_tool]

    state["tavily_report"] = tavily_report
    state["serp_report"] = serp_report
    state["report"] = {
        "structure": template_name,
        "content": final_report,
        "citations": filtered_sources,
        "dual_search": True,
        "winning_tool": winning_tool
    }

    logger.info(
        "Graph",
        f"SYNTHESIZE - Winner: {winning_tool} with {len(final_report)} chars",
        force_flush=True
    )


def _execute_single_synthesis(
    state: dict[str, Any],
    llm,
    query: str,
    sources: list[dict[str, Any]],
    template_name: str,
    is_gemini: bool
) -> None:
    logger.info("Graph", "SYNTHESIZE - Generating single report...", force_flush=True)

    limited_sources = sources
    if is_gemini and len(sources) > GEMINI_MAX_SOURCES_LIMIT:
        logger.info(
            "Graph",
            f"SYNTHESIZE - Limiting sources from {len(sources)} to {GEMINI_MAX_SOURCES_LIMIT} for Gemini",
            force_flush=True
        )
        limited_sources = sources[:GEMINI_MAX_SOURCES_LIMIT]

    sources_text = "\n".join([
        f"[{s['id']}] {s['title']} — {s['url']}"
        for s in limited_sources
    ])

    template_text = REPORT_TEMPLATES[template_name]
    template_text = add_provider_specific_instructions(template_text, is_gemini, template_name)

    system_prompt = f"{template_text}\nOnly cite using the numeric indices from SOURCES."
    user_prompt = f"QUERY:\n{query}\n\nSOURCES:\n{sources_text}"

    try:
        response = invoke_llm_safe(
            llm,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            is_gemini=is_gemini
        )

        state["report"] = {
            "structure": template_name,
            "content": response.content,
            "citations": limited_sources,
            "dual_search": False
        }

        logger.info(
            "Graph",
            f"SYNTHESIZE - Report: {len(response.content)} chars",
            force_flush=True
        )

    except LLMTimeoutError:
        state["report"] = {
            "structure": template_name,
            "content": "Report generation timed out. Please try with fewer search queries or use OpenAI.",
            "citations": limited_sources,
            "dual_search": False
        }


def _synthesize_single_report(
    llm,
    query: str,
    sources: list[dict[str, Any]],
    template_name: str,
    source_filter: str,
    is_gemini: bool
) -> str:
    filtered_sources = [s for s in sources if s.get("source") == source_filter]

    if not filtered_sources:
        return None

    if is_gemini and len(filtered_sources) > GEMINI_MAX_SOURCES_LIMIT:
        logger.info(
            "Graph",
            f"SYNTHESIZE - Limiting sources from {len(filtered_sources)} to {GEMINI_MAX_SOURCES_LIMIT} for Gemini",
            force_flush=True
        )
        filtered_sources = filtered_sources[:GEMINI_MAX_SOURCES_LIMIT]

    sources_text = "\n".join([
        f"[{s['id']}] {s['title']} — {s['url']} (from {s.get('source', 'unknown')})"
        for s in filtered_sources
    ])

    template_text = REPORT_TEMPLATES[template_name]
    template_text = add_provider_specific_instructions(template_text, is_gemini, template_name)

    system_prompt = f"{template_text}\nOnly cite using the numeric indices from SOURCES."
    user_prompt = f"QUERY:\n{query}\n\nSOURCES:\n{sources_text}"

    try:
        response = invoke_llm_safe(
            llm,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            is_gemini=is_gemini
        )

        content = response.content

        if template_name == TEMPLATE_TWO_COLUMN and is_gemini:
            content = _extract_table_from_response(content)

        return content

    except LLMTimeoutError:
        logger.error(
            "Graph",
            "SYNTHESIZE - Timeout occurred during synthesis",
            force_flush=True
        )
        return "Report generation timed out. Please try with fewer search queries or use OpenAI."


def _select_best_report(
    llm,
    query: str,
    tavily_report: str,
    serp_report: str,
    is_gemini: bool
) -> tuple[str, str]:
    logger.info(
        "Graph",
        "SYNTHESIZE - Asking LLM to select the BEST report...",
        force_flush=True
    )

    comparison_prompt = f"""You are a research quality evaluator. You have two research reports on the same topic from different search sources.

QUERY: {query}

TAVILY REPORT:
{tavily_report}

---

SERPAPI REPORT:
{serp_report}

---

Your task: Analyze both reports and select the BETTER one. Consider:
- Comprehensiveness and depth of information
- Source quality and credibility
- Factual accuracy and specificity
- Direct relevance to the query
- Clarity and structure

Respond with ONLY ONE of these exact phrases, nothing else:
- "TAVILY" if the Tavily report is better
- "SERPAPI" if the SerpAPI report is better

Your choice:"""

    try:
        choice_response = invoke_llm_safe(
            llm,
            [{"role": "user", "content": comparison_prompt}],
            is_gemini=is_gemini
        )
        choice = choice_response.content.strip().upper()
    except LLMTimeoutError:
        logger.warning(
            "Graph",
            "SYNTHESIZE - Comparison timed out, defaulting to Tavily",
            force_flush=True
        )
        choice = "TAVILY"

    logger.info("Graph", f"SYNTHESIZE - LLM chose: {choice}", force_flush=True)

    if "SERPAPI" in choice:
        return SEARCH_TOOL_SERP, serp_report
    else:
        return SEARCH_TOOL_TAVILY, tavily_report


def _extract_table_from_response(content: str) -> str:
    lines = content.split('\n')
    table_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Check if this line is part of a table (contains |)
        if '|' in stripped:
            in_table = True
            table_lines.append(line)
        elif in_table and not stripped:
            # Empty line after table - might be end of table
            continue
        elif in_table and stripped:
            # Non-table line after we were in a table - table ended
            break

    if table_lines:
        result = '\n'.join(table_lines)
        logger.info(
            "Graph",
            f"Extracted table with {len(table_lines)} lines from {len(lines)} total lines",
            force_flush=True
        )
        return result

    logger.warning(
        "Graph",
        "WARNING: No table found in response, returning original content",
        force_flush=True
    )
    return content