from typing import Literal, Any, Optional
from pydantic import BaseModel, Field

from constants import (
    TEMPLATE_BULLET_SUMMARY,
    TEMPLATE_TWO_COLUMN,
    TEMPLATE_DETAILED_REPORT,
    PROVIDER_OPENAI,
    PROVIDER_GEMINI,
)


Provider = Literal["openai", "gemini"]
"""Valid LLM provider names."""

TemplateName = Literal["bullet_summary", "two_column", "detailed_report"]
"""Valid report template names."""

MessageRole = Literal["user", "assistant", "tool"]
"""Valid message roles in conversation history."""

StreamEventType = Literal["token", "status", "done", "log", "plan", "sources", "progress", "error"]
"""Valid server-sent event types."""


class UserMessage(BaseModel):

    role: MessageRole = "user"
    content: str

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What are the latest trends in AI?"
            }
        }


class RunConfig(BaseModel):

    provider: Provider = PROVIDER_OPENAI
    model: Optional[str] = None
    template: TemplateName = TEMPLATE_BULLET_SUMMARY
    search_budget: int = Field(default=4, ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "template": "bullet_summary",
                "search_budget": 4
            }
        }


class RunRequest(BaseModel):

    query: str = Field(..., min_length=1, description="The research query")
    messages: list[UserMessage] = Field(default_factory=list)
    config: RunConfig = RunConfig()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the latest breakthroughs in quantum computing?",
                "messages": [],
                "config": {
                    "provider": "openai",
                    "template": "bullet_summary",
                    "search_budget": 4
                }
            }
        }


class Source(BaseModel):

    id: int
    title: str
    url: str
    snippet: str
    query: Optional[str] = None
    source: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "title": "Quantum Computing Breakthrough 2024",
                "url": "https://example.com/quantum-breakthrough",
                "snippet": "Researchers announced a major advancement...",
                "query": "quantum computing breakthroughs 2024",
                "source": "Tavily"
            }
        }


class Report(BaseModel):

    structure: TemplateName
    content: str
    citations: list[dict[str, Any]]
    dual_search: bool = False
    winning_tool: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "structure": "bullet_summary",
                "content": "## TL;DR\n\n- Key finding 1 [1]\n- Key finding 2 [2]",
                "citations": [
                    {
                        "id": 1,
                        "title": "Source Title",
                        "url": "https://example.com"
                    }
                ],
                "dual_search": True,
                "winning_tool": "Tavily"
            }
        }


class StreamChunk(BaseModel):

    event: StreamEventType = "token"
    data: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "event": "status",
                "data": {"phase": "searching"}
            }
        }


class ResearchState(BaseModel):

    query: str
    config: dict[str, Any]
    messages: list[dict[str, Any]]
    plan: str = ""
    search_results: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    report: Optional[dict[str, Any]] = None
    tavily_report: Optional[str] = None
    serp_report: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True