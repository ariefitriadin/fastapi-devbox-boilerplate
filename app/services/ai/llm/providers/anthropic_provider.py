"""
Anthropic Provider - Claude 3.5 Sonnet, Opus, Haiku
Best for reasoning, long context, and safety
"""

import logging
import time
from typing import AsyncIterator, List, Optional

from anthropic import AnthropicError, AsyncAnthropic

from app.core.ai_config import AISettings

from .base import BaseLLMProvider, LLMMessage, LLMResponse, LLMUsageMetrics

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Provider (Claude models)

    Pricing (as of 2024):
    - claude-3-5-sonnet: $3/1M input, $15/1M output (RECOMMENDED)
    - claude-3-opus: $15/1M input, $75/1M output (Most capable)
    - claude-3-haiku: $0.25/1M input, $1.25/1M output (Fastest, cheapest)

    Features:
    - 200K context window (huge!)
    - Strong reasoning capabilities
    - Better at following instructions
    - More cautious with sensitive topics
    """

    PRICING = {
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def __init__(self, settings: AISettings):
        super().__init__(settings)

        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Get one at: https://console.anthropic.com/"
            )

        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        logger.info("Anthropic provider initialized")

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self.settings.anthropic_model

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
        """Generate completion using Claude"""
        start_time = time.time()
        model = model or self.default_model
        temperature = temperature or self.settings.openai_temperature
        max_tokens = max_tokens or self.settings.openai_max_tokens

        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            content = response.content[0].text
            latency_ms = (time.time() - start_time) * 1000

            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens

            cost = self.calculate_cost(prompt_tokens, completion_tokens, model)

            usage = LLMUsageMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                model=model,
                provider=self.provider_name,
            )

            self._track_usage(usage)

            return LLMResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage=usage,
                metadata={"stop_reason": response.stop_reason, "id": response.id},
            )

        except AnthropicError as e:
            logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Anthropic request failed: {str(e)}")

    async def chat(
        self,
        messages: List[LLMMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Multi-turn chat with Claude"""
        start_time = time.time()
        model = model or self.default_model
        temperature = temperature or self.settings.openai_temperature
        max_tokens = max_tokens or self.settings.openai_max_tokens

        # Extract system message if present
        system_prompt = ""
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                chat_messages.append(msg.to_dict())

        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=chat_messages,
                **kwargs,
            )

            content = response.content[0].text
            latency_ms = (time.time() - start_time) * 1000

            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens

            cost = self.calculate_cost(prompt_tokens, completion_tokens, model)

            usage = LLMUsageMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                model=model,
                provider=self.provider_name,
            )

            self._track_usage(usage)

            return LLMResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage=usage,
                metadata={"stop_reason": response.stop_reason},
            )

        except AnthropicError as e:
            logger.error(f"Anthropic chat error: {e}")
            raise RuntimeError(f"Anthropic chat failed: {str(e)}")

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
        """Stream tokens from Claude"""
        model = model or self.default_model
        temperature = temperature or self.settings.openai_temperature
        max_tokens = max_tokens or self.settings.openai_max_tokens

        try:
            async with self.client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except AnthropicError as e:
            logger.error(f"Anthropic stream error: {e}")
            raise RuntimeError(f"Anthropic stream failed: {str(e)}")

    def calculate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: str
    ) -> float:
        """Calculate cost based on Anthropic pricing"""
        # Get pricing for model (fallback to Sonnet pricing)
        pricing = self.PRICING.get(model, self.PRICING["claude-3-5-sonnet-20240620"])

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost
