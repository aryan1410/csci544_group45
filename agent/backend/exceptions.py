class DeepResearchError(Exception):

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(DeepResearchError):
    pass


class APIKeyError(ConfigurationError):
    pass


class ProviderError(DeepResearchError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class SearchError(DeepResearchError):
    pass


class SearchToolError(SearchError):
    pass


class LLMTimeoutError(ProviderError):
    pass


class SynthesisError(DeepResearchError):
    pass


class ValidationError(DeepResearchError):
    pass