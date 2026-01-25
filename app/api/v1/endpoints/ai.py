"""
AI Endpoints - Chat, Generation, and Streaming
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.ai_config import LLMProvider
from app.services.ai import AIServiceFactory, get_ai_factory
from app.services.ai.llm.providers import LLMMessage

router = APIRouter(prefix="/ai", tags=["AI"])


# ============================================
# Request/Response Models
# ============================================


class GenerateRequest(BaseModel):
    """Request for text generation"""

    prompt: str = Field(..., description="Input prompt for generation")
    system_prompt: Optional[str] = Field(
        None, description="System prompt to guide behavior"
    )
    provider: Optional[str] = Field(
        None, description="Override provider (openai, anthropic, ollama)"
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Randomness (0.0-2.0)"
    )
    max_tokens: Optional[int] = Field(
        None, gt=0, le=8000, description="Maximum tokens to generate"
    )


class ChatMessageModel(BaseModel):
    """Single chat message"""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for multi-turn chat"""

    messages: List[ChatMessageModel] = Field(
        ..., min_length=1, description="Conversation history"
    )
    provider: Optional[str] = Field(None, description="Override provider")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0, le=8000)


class UsageMetrics(BaseModel):
    """Token usage and cost information"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    model: str
    provider: str


class GenerateResponse(BaseModel):
    """Response from generation/chat endpoints"""

    content: str
    model: str
    provider: str
    usage: UsageMetrics


class ProviderInfo(BaseModel):
    """Information about available providers"""

    name: str
    available: bool
    description: str
    cost: str


class ProvidersResponse(BaseModel):
    """List of available providers"""

    default_provider: str
    available_providers: List[ProviderInfo]


# ============================================
# Endpoints
# ============================================


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest, factory: AIServiceFactory = Depends(get_ai_factory)
):
    """
    Generate AI completion from a prompt

    **Example Request:**
    ```json
    {
        "prompt": "Explain what FastAPI is in 2 sentences",
        "system_prompt": "You are a helpful assistant",
        "temperature": 0.7,
        "max_tokens": 200
    }
    ```

    **Example Response:**
    ```json
    {
        "content": "FastAPI is a modern, high-performance web framework...",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 42,
            "total_tokens": 57,
            "cost_usd": 0.0001,
            "latency_ms": 823,
            "model": "gpt-4o-mini",
            "provider": "openai"
        }
    }
    ```
    """
    try:
        # Get provider
        provider = None
        if request.provider:
            try:
                provider = LLMProvider(request.provider)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid provider: {request.provider}. "
                    f"Must be one of: openai, anthropic, ollama",
                )

        # Create LLM client
        llm = factory.create_llm_client(provider=provider)

        # Generate response
        response = await llm.generate(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return GenerateResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            usage=UsageMetrics(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cost_usd=response.usage.cost_usd,
                latency_ms=response.usage.latency_ms,
                model=response.usage.model,
                provider=response.usage.provider,
            ),
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.post("/chat", response_model=GenerateResponse)
async def chat_completion(
    request: ChatRequest, factory: AIServiceFactory = Depends(get_ai_factory)
):
    """
    Multi-turn chat conversation

    **Example Request:**
    ```json
    {
        "messages": [
            {"role": "system", "content": "You are a Python expert"},
            {"role": "user", "content": "What is FastAPI?"},
            {"role": "assistant", "content": "FastAPI is a web framework..."},
            {"role": "user", "content": "Show me an example"}
        ],
        "temperature": 0.7
    }
    ```
    """
    try:
        # Validate roles
        valid_roles = {"system", "user", "assistant"}
        for msg in request.messages:
            if msg.role not in valid_roles:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid role: {msg.role}. Must be one of: {valid_roles}",
                )

        # Get provider
        provider = None
        if request.provider:
            try:
                provider = LLMProvider(request.provider)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid provider: {request.provider}",
                )

        # Convert to LLMMessage format
        messages = [
            LLMMessage(role=msg.role, content=msg.content) for msg in request.messages
        ]

        # Create LLM client
        llm = factory.create_llm_client(provider=provider)

        # Generate chat response
        response = await llm.chat(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return GenerateResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            usage=UsageMetrics(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cost_usd=response.usage.cost_usd,
                latency_ms=response.usage.latency_ms,
                model=response.usage.model,
                provider=response.usage.provider,
            ),
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers(factory: AIServiceFactory = Depends(get_ai_factory)):
    """
    List available AI providers and their status

    Returns information about which providers are configured and available.
    """
    from app.core.ai_config import ai_settings

    available = factory.get_available_providers()

    providers_info = [
        ProviderInfo(
            name="ollama",
            available="ollama" in available,
            description="Local LLM inference (Llama3, Mistral, etc.)",
            cost="FREE",
        ),
        ProviderInfo(
            name="openai",
            available="openai" in available,
            description="GPT-4o, GPT-4o-mini, GPT-3.5",
            cost="$0.15-$15/1M tokens",
        ),
        ProviderInfo(
            name="anthropic",
            available="anthropic" in available,
            description="Claude 3.5 Sonnet, Opus, Haiku",
            cost="$0.25-$75/1M tokens",
        ),
    ]

    return ProvidersResponse(
        default_provider=ai_settings.default_llm_provider.value,
        available_providers=providers_info,
    )


@router.get("/health")
async def ai_health_check(factory: AIServiceFactory = Depends(get_ai_factory)):
    """
    Check AI services health

    Tests connectivity to configured providers.
    """
    from app.core.ai_config import ai_settings

    health = {
        "status": "healthy",
        "default_provider": ai_settings.default_llm_provider.value,
        "available_providers": factory.get_available_providers(),
    }

    # Try to ping default provider
    try:
        llm = factory.create_llm_client()
        health["default_provider_status"] = "connected"
        health["default_model"] = llm.default_model
    except Exception as e:
        health["status"] = "degraded"
        health["default_provider_status"] = "error"
        health["error"] = str(e)

    return health
