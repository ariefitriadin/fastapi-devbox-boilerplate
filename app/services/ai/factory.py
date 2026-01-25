"""
Factory Pattern for AI Service Creation
Supports dynamic provider switching at runtime
"""

from typing import Optional

from app.core.ai_config import AISettings, EmbeddingProvider, LLMProvider

from .llm.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    OllamaProvider,
    OpenAIProvider,
)


class AIServiceFactory:
    """
    Factory for creating AI service instances with proper provider

    Usage:
        factory = AIServiceFactory(ai_settings)
        llm = factory.create_llm_client()
        llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)
    """

    def __init__(self, settings: AISettings):
        self.settings = settings
        self._llm_cache: dict[str, BaseLLMProvider] = {}

    def create_llm_client(
        self, provider: Optional[LLMProvider] = None
    ) -> BaseLLMProvider:
        """
        Create LLM client for specified provider

        Args:
            provider: Override default provider (useful for A/B testing)

        Returns:
            LLM client implementing BaseLLMProvider protocol

        Example:
            # Use default provider from settings
            llm = factory.create_llm_client()

            # Override to use Ollama for free local inference
            llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)

            # Generate text
            response = await llm.generate("Explain AI")
            print(response.content)
            print(f"Cost: ${response.usage.cost_usd:.4f}")
        """
        provider = provider or self.settings.default_llm_provider

        # Return cached instance if exists
        cache_key = provider.value
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        # Create new instance based on provider
        if provider == LLMProvider.OPENAI:
            client = OpenAIProvider(self.settings)

        elif provider == LLMProvider.ANTHROPIC:
            client = AnthropicProvider(self.settings)

        elif provider == LLMProvider.OLLAMA:
            client = OllamaProvider(self.settings)

        elif provider == LLMProvider.AZURE_OPENAI:
            # TODO: Implement Azure OpenAI provider
            raise NotImplementedError("Azure OpenAI provider not yet implemented")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Cache and return
        self._llm_cache[cache_key] = client
        return client

    def get_available_providers(self) -> list[str]:
        """
        Get list of available providers based on configuration

        Returns:
            List of provider names that have valid API keys/configuration
        """
        available = []

        # Check Ollama (always available if running)
        available.append("ollama")

        # Check OpenAI
        if self.settings.openai_api_key:
            available.append("openai")

        # Check Anthropic
        if self.settings.anthropic_api_key:
            available.append("anthropic")

        # Check Azure OpenAI
        if self.settings.azure_openai_api_key and self.settings.azure_openai_endpoint:
            available.append("azure-openai")

        return available

    def get_cheapest_provider(self) -> LLMProvider:
        """
        Get the cheapest available provider

        Returns:
            LLMProvider enum for the most cost-effective option
        """
        # Ollama is always free if available
        return LLMProvider.OLLAMA

    def create_embedding_client(self, provider: Optional[EmbeddingProvider] = None):
        """
        Create embedding client for specified provider

        TODO: Implement embedding providers
        """
        provider = provider or self.settings.embedding_provider
        raise NotImplementedError("Embedding providers not yet implemented")

    def create_vision_client(self):
        """
        Create YOLO-based computer vision client

        TODO: Implement vision service
        """
        raise NotImplementedError("Vision service not yet implemented")

    def clear_cache(self):
        """Clear all cached provider instances"""
        self._llm_cache.clear()


# Global factory instance
_factory: Optional[AIServiceFactory] = None


def get_ai_factory() -> AIServiceFactory:
    """
    Dependency injection helper for FastAPI

    Usage in FastAPI endpoints:
        from fastapi import Depends
        from app.services.ai.factory import get_ai_factory

        @app.post("/ai/generate")
        async def generate_text(
            prompt: str,
            factory: AIServiceFactory = Depends(get_ai_factory)
        ):
            llm = factory.create_llm_client()
            response = await llm.generate(prompt)
            return response.to_dict()
    """
    global _factory
    if _factory is None:
        from app.core.ai_config import ai_settings

        _factory = AIServiceFactory(ai_settings)
    return _factory
