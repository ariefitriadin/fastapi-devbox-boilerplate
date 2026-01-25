"""
OpenAI Provider - GPT-4, GPT-3.5, etc.
Industry standard, reliable, best quality
"""

import logging
import time
from typing import AsyncIterator, List, Optional

from openai import AsyncOpenAI, OpenAIError

from app.core.ai_config import AISettings

from .base import BaseLLMProvider, LLMMessage, LLMResponse, LLMUsageMetrics

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI Provider (GPT-4, GPT-3.5-turbo, GPT-4o-mini)

    Pricing (as of 2024):
    - gpt-4o-mini: $0.15/1M input, $0.60/1M output (BEST VALUE)
    - gpt-4o: $5.00/1M input, $15.00/1M output
    - gpt-4-turbo: $10/1M input, $30/1M output
    """

    # Token pricing per 1M tokens
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, settings: AISettings):
        super().__init__(settings)

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Get one at: https://platform.openai.com/api-keys"
            )

        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info("OpenAI provider initialized")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self.settings.openai_model

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
        """Generate completion using OpenAI"""
        start_time = time.time()
        model = model or self.default_model
        temperature = temperature or self.settings.openai_temperature
        max_tokens = max_tokens or self.settings.openai_max_tokens

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            content = response.choices[0].message.content
            latency_ms = (time.time() - start_time) * 1000

            # Extract usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

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
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "id": response.id,
                },
            )

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI request failed: {str(e)}")

    async def chat(
        self,
        messages: List[LLMMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Multi-turn chat with OpenAI"""
        start_time = time.time()
        model = model or self.default_model
        temperature = temperature or self.settings.openai_temperature
        max_tokens = max_tokens or self.settings.openai_max_tokens

        # Convert to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            content = response.choices[0].message.content
            latency_ms = (time.time() - start_time) * 1000

            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

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
                metadata={"finish_reason": response.choices[0].finish_reason},
            )

        except OpenAIError as e:
            logger.error(f"OpenAI chat error: {e}")
            raise RuntimeError(f"OpenAI chat failed: {str(e)}")

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
        """Stream tokens from OpenAI"""
        model = model or self.default_model
        temperature = temperature or self.settings.openai_temperature
        max_tokens = max_tokens or self.settings.openai_max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIError as e:
            logger.error(f"OpenAI stream error: {e}")
            raise RuntimeError(f"OpenAI stream failed: {str(e)}")

    def calculate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: str
    ) -> float:
        """Calculate cost based on OpenAI pricing"""
        # Get pricing for model (fallback to gpt-4o-mini pricing)
        pricing = self.PRICING.get(model, self.PRICING["gpt-4o-mini"])

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost
