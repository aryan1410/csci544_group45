import os
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from constants import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MAX_SEARCHES,
    GEMINI_MAX_OUTPUT_TOKENS_DEFAULT,
    GEMINI_TIMEOUT_SECONDS_DEFAULT,
    GEMINI_REQUEST_TIMEOUT_DEFAULT,
    GEMINI_MAX_RETRIES_DEFAULT,
    GEMINI_MAX_SEARCHES_DEFAULT,
    PROVIDER_OPENAI,
    PROVIDER_GEMINI,
)
from exceptions import APIKeyError

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))


Provider = Literal["openai", "gemini"]


class Settings(BaseSettings):

    OPENAI_API_KEY: Optional[str] = None
    """OpenAI API key for GPT models."""

    GEMINI_API_KEY: Optional[str] = None
    """Google Gemini API key."""

    TAVILY_API_KEY: Optional[str] = None
    """Tavily search API key (required for search functionality)."""

    SERP_API_KEY: Optional[str] = None
    """SerpAPI key for additional search functionality (optional)."""

    MODEL: str = DEFAULT_OPENAI_MODEL
    """Default model to use for OpenAI."""

    MAX_MESSAGES: int = DEFAULT_MAX_MESSAGES
    """Maximum number of messages to keep in rolling memory buffer."""

    MAX_SEARCHES: int = DEFAULT_MAX_SEARCHES
    """Maximum number of search queries allowed per request."""

    USE_DUAL_SEARCH: bool = True
    """Whether to enable dual search (Tavily + SerpAPI) mode."""

    GEMINI_MAX_OUTPUT_TOKENS: int = GEMINI_MAX_OUTPUT_TOKENS_DEFAULT
    """Maximum number of output tokens for Gemini responses."""

    GEMINI_TIMEOUT_SECONDS: int = GEMINI_TIMEOUT_SECONDS_DEFAULT
    """Timeout in seconds for Gemini API calls."""

    GEMINI_REQUEST_TIMEOUT: int = GEMINI_REQUEST_TIMEOUT_DEFAULT
    """Overall request timeout for Gemini operations."""

    GEMINI_MAX_RETRIES: int = GEMINI_MAX_RETRIES_DEFAULT
    """Maximum number of retries for failed Gemini requests."""

    GEMINI_MAX_SEARCHES: int = GEMINI_MAX_SEARCHES_DEFAULT
    """Maximum number of searches allowed when using Gemini (quota management)."""


    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_available_provider(self, requested_provider: Provider) -> Provider:
        if requested_provider == PROVIDER_GEMINI:
            if not self.GEMINI_API_KEY:
                print(f"[Settings] GEMINI_API_KEY not found, falling back to {PROVIDER_OPENAI}")
                if not self.OPENAI_API_KEY:
                    raise APIKeyError(
                        f"Cannot use {PROVIDER_GEMINI} (no GEMINI_API_KEY) and cannot fallback to "
                        f"{PROVIDER_OPENAI} (no OPENAI_API_KEY). Please provide at least one API key."
                    )
                return PROVIDER_OPENAI
            return PROVIDER_GEMINI

        if not self.OPENAI_API_KEY:
            raise APIKeyError(
                "OPENAI_API_KEY is required. Please add it to your .env file. "
                "Get one at: https://platform.openai.com/api-keys"
            )
        return PROVIDER_OPENAI

    def validate_search_requirements(self) -> None:
        if not self.TAVILY_API_KEY:
            raise APIKeyError(
                "TAVILY_API_KEY is required for search functionality. "
                "Please add it to your .env file. Get one at: https://tavily.com"
            )

    @property
    def has_serp_api(self) -> bool:
        return bool(self.SERP_API_KEY)

    @property
    def has_gemini(self) -> bool:
        return bool(self.GEMINI_API_KEY)

    @property
    def has_tavily(self) -> bool:
        return bool(self.TAVILY_API_KEY)

    @property
    def can_use_dual_search(self) -> bool:
        return self.USE_DUAL_SEARCH and self.has_serp_api and self.has_tavily


settings = Settings()