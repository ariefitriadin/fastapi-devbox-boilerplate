"""
Example: Using Ollama Provider for Local LLM Inference
Prerequisites:
1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
2. Start Ollama: ollama serve
3. Pull a model: ollama pull llama3.1:8b
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ai_config import LLMProvider, ai_settings
from app.services.ai.factory import AIServiceFactory
from app.services.ai.llm.providers import LLMMessage


async def test_simple_generation():
    """Test 1: Simple text generation"""
    print("=" * 60)
    print("TEST 1: Simple Text Generation")
    print("=" * 60)

    factory = AIServiceFactory(ai_settings)
    llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)

    print(f"Using provider: {llm.provider_name}")
    print(f"Model: {llm.default_model}\n")

    prompt = "Explain what FastAPI is in 2 sentences."
    print(f"Prompt: {prompt}\n")

    response = await llm.generate(prompt)

    print(f"Response: {response.content}\n")
    print(f"Tokens: {response.usage.total_tokens}")
    print(f"Cost: ${response.usage.cost_usd:.4f} (FREE!)")
    print(f"Latency: {response.usage.latency_ms:.0f}ms")
    print()


async def test_system_prompt():
    """Test 2: Using system prompts"""
    print("=" * 60)
    print("TEST 2: System Prompt")
    print("=" * 60)

    factory = AIServiceFactory(ai_settings)
    llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)

    system_prompt = "You are a Python expert. Answer concisely."
    prompt = "What's the difference between async and sync?"

    print(f"System: {system_prompt}")
    print(f"Prompt: {prompt}\n")

    response = await llm.generate(
        prompt,
        system_prompt=system_prompt,
        temperature=0.5,  # Lower temperature for more focused responses
    )

    print(f"Response: {response.content}\n")
    print(f"Latency: {response.usage.latency_ms:.0f}ms")
    print()


async def test_multi_turn_chat():
    """Test 3: Multi-turn conversation"""
    print("=" * 60)
    print("TEST 3: Multi-turn Chat Conversation")
    print("=" * 60)

    factory = AIServiceFactory(ai_settings)
    llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)

    messages = [
        LLMMessage(role="system", content="You are a helpful AI assistant."),
        LLMMessage(role="user", content="What is machine learning?"),
        LLMMessage(
            role="assistant",
            content="Machine learning is a subset of AI where computers learn from data without explicit programming.",
        ),
        LLMMessage(role="user", content="Give me a simple example."),
    ]

    print("Conversation history:")
    for msg in messages:
        print(f"  [{msg.role}]: {msg.content}")
    print()

    response = await llm.chat(messages)

    print(f"[assistant]: {response.content}\n")
    print(f"Total cost: ${response.usage.cost_usd:.4f}")
    print()


async def test_streaming():
    """Test 4: Streaming responses"""
    print("=" * 60)
    print("TEST 4: Streaming Response")
    print("=" * 60)

    factory = AIServiceFactory(ai_settings)
    llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)

    prompt = "Write a haiku about coding."
    print(f"Prompt: {prompt}\n")
    print("Streaming response: ", end="", flush=True)

    async for token in llm.stream(prompt):
        print(token, end="", flush=True)

    print("\n")


async def test_cost_tracking():
    """Test 5: Cost tracking across multiple requests"""
    print("=" * 60)
    print("TEST 5: Cost Tracking")
    print("=" * 60)

    factory = AIServiceFactory(ai_settings)
    llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)

    prompts = ["What is Python?", "What is JavaScript?", "What is Rust?"]

    for i, prompt in enumerate(prompts, 1):
        response = await llm.generate(prompt, max_tokens=50)
        print(f"{i}. {prompt[:30]}... → {response.usage.total_tokens} tokens")

    print(f"\nTotal requests: {llm.get_request_count()}")
    print(f"Total cost: ${llm.get_total_cost():.4f}")
    print()


async def test_provider_comparison():
    """Test 6: Compare available providers"""
    print("=" * 60)
    print("TEST 6: Available Providers")
    print("=" * 60)

    factory = AIServiceFactory(ai_settings)
    available = factory.get_available_providers()

    print(f"Configured providers: {', '.join(available)}")
    print(f"Default provider: {ai_settings.default_llm_provider}")
    print(f"Cheapest provider: {factory.get_cheapest_provider()}")
    print()


async def test_error_handling():
    """Test 7: Error handling when Ollama is not running"""
    print("=" * 60)
    print("TEST 7: Error Handling")
    print("=" * 60)

    # Temporarily use wrong URL to test error handling
    from app.core.ai_config import AISettings

    bad_settings = AISettings(
        ollama_base_url="http://localhost:99999",  # Wrong port
        default_llm_provider=LLMProvider.OLLAMA,
    )

    factory = AIServiceFactory(bad_settings)
    llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)

    try:
        response = await llm.generate("Hello")
        print(f"Response: {response.content}")
    except RuntimeError as e:
        print(f"✓ Caught expected error: {str(e)[:80]}...")
    print()


async def main():
    """Run all tests"""
    print("\n")
    print("🦙 OLLAMA PROVIDER TESTS")
    print("=" * 60)
    print()

    try:
        await test_simple_generation()
        await test_system_prompt()
        await test_multi_turn_chat()
        await test_streaming()
        await test_cost_tracking()
        await test_provider_comparison()
        await test_error_handling()

        print("=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure:")
        print("1. Ollama is installed: curl -fsSL https://ollama.com/install.sh | sh")
        print("2. Ollama is running: ollama serve")
        print("3. Model is pulled: ollama pull llama3.1:8b")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
