"""
Ollama Provider - Local LLM Inference
Free, private, runs on your machine
"""

import json
import logging
import time
from typing import AsyncIterator, List, Optional

import httpx

from app.core.ai_config import AISettings

from .base import BaseLLMProvider, LLMMessage, LLMResponse, LLMUsageMetrics

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama Provider for local LLM inference

    Features:
    - 100% free (no API costs)
    - Private (data stays local)
    - Supports Llama3, Mistral, Gemma, etc.
    - No rate limits

    Prerequisites:
    - Ollama must be installed and running
    - Install: curl -fsSL https://ollama.com/install.sh | sh
    - Pull model: ollama pull llama3.1:8b
    """

    def __init__(self, settings: AISettings):
        super().__init__(settings)
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"Ollama provider initialized: {self.base_url}")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self.settings.ollama_model

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
        Generate completion using Ollama

        Example:
            provider = OllamaProvider(settings)
            response = await provider.generate(
                "Explain quantum computing",
                model="llama3.1:8b",
                temperature=0.7
            )
            print(response.content)
        """
        start_time = time.time()
        model = model or self.default_model
        temperature = temperature or self.settings.openai_temperature

        # Build request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            # Call Ollama API
            response = await self.client.post(
                f"{self.base_url}/api/generate", json=payload
            )
            response.raise_for_status()
            data = response.json()

            # Extract response
            content = data.get("response", "")

            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000

            # Ollama doesn't return exact token counts, estimate
            prompt_tokens = self._estimate_tokens(prompt)
            completion_tokens = self._estimate_tokens(content)

            usage = LLMUsageMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=0.0,  # Ollama is free!
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
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                },
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise RuntimeError(
                f"Ollama request failed: {e.response.text}. "
                "Is Ollama running? Try: ollama serve"
            )
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )

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
        Multi-turn chat with Ollama

        Example:
            messages = [
                LLMMessage(role="system", content="You are a helpful assistant"),
                LLMMessage(role="user", content="What is Python?"),
                LLMMessage(role="assistant", content="Python is a programming language"),
                LLMMessage(role="user", content="Tell me more")
            ]
            response = await provider.chat(messages)
        """
        start_time = time.time()
        model = model or self.default_model
        temperature = temperature or self.settings.openai_temperature

        # Convert messages to Ollama format
        ollama_messages = [msg.to_dict() for msg in messages]

        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            content = data.get("message", {}).get("content", "")
            latency_ms = (time.time() - start_time) * 1000

            # Estimate tokens
            prompt_text = " ".join(msg.content for msg in messages)
            prompt_tokens = self._estimate_tokens(prompt_text)
            completion_tokens = self._estimate_tokens(content)

            usage = LLMUsageMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=0.0,
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
                metadata=data,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama chat error: {e}")
            raise RuntimeError(f"Ollama chat failed: {e.response.text}")

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
        Stream completion tokens in real-time

        Example:
            async for token in provider.stream("Write a story"):
                print(token, end="", flush=True)
        """
        model = model or self.default_model
        temperature = temperature or self.settings.openai_temperature

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }

        if system_prompt:
            payload["system"] = system_prompt

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama stream error: {e}")
            raise RuntimeError(f"Ollama stream failed: {e.response.text}")

    def calculate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: str
    ) -> float:
        """Ollama is always free!"""
        return 0.0

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (1 token ≈ 4 characters)
        Ollama doesn't return exact counts, so we approximate
        """
        return len(text) // 4

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
