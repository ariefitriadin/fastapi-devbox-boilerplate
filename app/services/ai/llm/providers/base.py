"""
Abstract Base Class for all LLM providers
Ensures consistent interface across OpenAI, Anthropic, Ollama, etc.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from app.core.ai_config import AISettings

logger = logging.getLogger(__name__)


@dataclass
class LLMUsageMetrics:
    """Token usage and cost tracking"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LLMResponse:
    """Standardized LLM response"""

    content: str
    model: str
    provider: str
    usage: LLMUsageMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class LLMMessage:
    """Chat message format"""

    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None

    def to_dict(self) -> dict:
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers

    All providers (OpenAI, Anthropic, Ollama, etc.) must implement these methods
    """

    def __init__(self, settings: AISettings):
        self.settings = settings
        self._total_cost = 0.0
        self._request_count = 0

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (e.g., 'openai', 'anthropic')"""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return default model name for this provider"""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion for a single prompt

        Args:
            prompt: User prompt/question
            model: Override default model
            temperature: Randomness (0.0-2.0)
            max_tokens: Max completion length
            system_prompt: System/instruction prompt
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with content and usage metrics
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Multi-turn chat completion

        Args:
            messages: List of conversation messages
            model: Override default model
            temperature: Randomness
            max_tokens: Max completion length
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with assistant's reply
        """
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream completion tokens as they arrive

        Args:
            Same as generate()

        Yields:
            Token strings as they're generated
        """
        pass

    @abstractmethod
    def calculate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: str
    ) -> float:
        """
        Calculate cost in USD for given token usage

        Args:
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            model: Model name

        Returns:
            Cost in USD
        """
        pass

    def get_total_cost(self) -> float:
        """Get cumulative cost for this provider instance"""
        return self._total_cost

    def get_request_count(self) -> int:
        """Get total requests made"""
        return self._request_count

    def _track_usage(self, usage: LLMUsageMetrics):
        """Internal method to track usage metrics"""
        self._total_cost += usage.cost_usd
        self._request_count += 1
        logger.info(
            f"{self.provider_name} request completed: "
            f"tokens={usage.total_tokens}, cost=${usage.cost_usd:.4f}, "
            f"latency={usage.latency_ms:.0f}ms"
        )
