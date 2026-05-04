from typing import Final

ALLOWED_ORIGINS: Final[list[str]] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://deep-research-33lv.onrender.com"
]
ALLOWED_ORIGIN_REGEX: Final[str] = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.0-flash-exp"

DEFAULT_TEMPERATURE: Final[float] = 0.2

GEMINI_MAX_OUTPUT_TOKENS_DEFAULT: Final[int] = 2048
GEMINI_TIMEOUT_SECONDS_DEFAULT: Final[int] = 30
GEMINI_REQUEST_TIMEOUT_DEFAULT: Final[int] = 45
GEMINI_MAX_RETRIES_DEFAULT: Final[int] = 1
GEMINI_MAX_SEARCHES_DEFAULT: Final[int] = 4
GEMINI_MAX_SOURCES_LIMIT: Final[int] = 10

DEFAULT_SEARCH_RESULTS: Final[int] = 5
MAX_SEARCH_RESULTS: Final[int] = 20
DEFAULT_MAX_SEARCHES: Final[int] = 12

DETAILED_QUERY_COUNT_OPENAI: Final[str] = "8-12"
DETAILED_QUERY_COUNT_GEMINI: Final[str] = "4-6"
SIMPLE_QUERY_COUNT_OPENAI: Final[str] = "3-6"
SIMPLE_QUERY_COUNT_GEMINI: Final[str] = "3-4"


DEFAULT_MAX_MESSAGES: Final[int] = 12


SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # Disable nginx buffering
}


PROGRESS_PLANNING: Final[int] = 10
PROGRESS_PLAN_COMPLETE: Final[int] = 33
PROGRESS_SEARCHING: Final[int] = 40
PROGRESS_SEARCH_COMPLETE: Final[int] = 66
PROGRESS_SYNTHESIZING: Final[int] = 75
PROGRESS_SYNTHESIS_COMPLETE: Final[int] = 90
PROGRESS_DONE: Final[int] = 100


TEMPLATE_BULLET_SUMMARY: Final[str] = "bullet_summary"
TEMPLATE_TWO_COLUMN: Final[str] = "two_column"
TEMPLATE_DETAILED_REPORT: Final[str] = "detailed_report"


PROVIDER_OPENAI: Final[str] = "openai"
PROVIDER_GEMINI: Final[str] = "gemini"


PHASE_PLANNING: Final[str] = "planning"
PHASE_SEARCHING: Final[str] = "searching"
PHASE_SYNTHESIZING: Final[str] = "synthesizing"
PHASE_DONE: Final[str] = "done"
PHASE_ERROR: Final[str] = "error"


SEARCH_TOOL_TAVILY: Final[str] = "Tavily"
SEARCH_TOOL_SERP: Final[str] = "SerpAPI"


PROGRESS_RESET_DELAY: Final[int] = 2