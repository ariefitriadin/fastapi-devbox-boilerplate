# 🚀 Quick Start: AI Integration with Ollama

Get AI capabilities running in **5 minutes** using free, local LLM inference.

## Prerequisites

- ✅ Python 3.13+ installed
- ✅ Devbox environment set up
- ✅ Basic FastAPI boilerplate running

## Step 1: Install Ollama (2 minutes)

### macOS / Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download installer from: https://ollama.com/download

### Verify Installation
```bash
ollama --version
```

## Step 2: Pull an LLM Model (3 minutes)

```bash
# Start Ollama server (keep this running)
ollama serve

# In a new terminal, pull a model
ollama pull llama3.1:8b

# Verify it works
ollama run llama3.1:8b "Hello, how are you?"
```

**Model Options:**
- `llama3.1:8b` - **Recommended** (4.7GB, best balance)
- `mistral:7b` - Faster, smaller (4.1GB)
- `gemma2:9b` - Google's model (5.4GB)
- `llama3.1:70b` - Most powerful (40GB, needs GPU)

## Step 3: Configure Environment

```bash
# Copy the environment template
cp env.ai.template .env

# Edit .env and ensure these lines are set:
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

## Step 4: Install Dependencies

```bash
# Enter devbox shell
devbox shell

# Install AI packages
poetry install
```

## Step 5: Test the Integration

### Option A: Run Test Script
```bash
poetry run python examples/test_ollama.py
```

**Expected Output:**
```
🦙 OLLAMA PROVIDER TESTS
============================================================

TEST 1: Simple Text Generation
============================================================
Using provider: ollama
Model: llama3.1:8b

Prompt: Explain what FastAPI is in 2 sentences.

Response: FastAPI is a modern, high-performance web framework...

Tokens: 157
Cost: $0.0000 (FREE!)
Latency: 1234ms
```

### Option B: Start API Server
```bash
# Start the FastAPI server
poetry run uvicorn app.main:app --reload --port 8000
```

**Test with cURL:**
```bash
# Simple generation
curl -X POST "http://localhost:8000/api/v1/ai/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain Python in one sentence",
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**Test with Browser:**
Open http://localhost:8000/docs and try the `/api/v1/ai/generate` endpoint.

## Step 6: Use in Your Code

```python
from app.services.ai import get_ai_factory
from app.core.ai_config import LLMProvider

# Create factory
factory = get_ai_factory()

# Get LLM client (uses Ollama by default)
llm = factory.create_llm_client()

# Generate text
response = await llm.generate("What is machine learning?")
print(response.content)
print(f"Cost: ${response.usage.cost_usd:.4f}")  # Always $0.00 with Ollama!
```

## Common Issues & Solutions

### ❌ "Cannot connect to Ollama"
```bash
# Solution: Make sure Ollama is running
ollama serve
```

### ❌ "Model not found: llama3.1:8b"
```bash
# Solution: Pull the model
ollama pull llama3.1:8b

# Check what models you have
ollama list
```

### ❌ "Slow response times"
**Solutions:**
- Use a smaller model: `ollama pull mistral:7b`
- Reduce max_tokens in requests
- Use GPU if available (automatic detection)

### ❌ "Out of memory"
**Solutions:**
- Use smaller model: `llama3.1:8b` instead of `llama3.1:70b`
- Close other applications
- Reduce `num_ctx` in Ollama settings

## Next Steps

### 1. Add Streaming Responses
```python
async for token in llm.stream("Write a story"):
    print(token, end="", flush=True)
```

### 2. Multi-turn Conversations
```python
from app.services.ai.llm.providers import LLMMessage

messages = [
    LLMMessage(role="system", content="You are a helpful assistant"),
    LLMMessage(role="user", content="What is Python?"),
    LLMMessage(role="assistant", content="Python is a programming language"),
    LLMMessage(role="user", content="Show me an example")
]

response = await llm.chat(messages)
```

### 3. Switch to Production Provider
When ready for production, switch to OpenAI:

```bash
# .env
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

No code changes needed! The same API works across all providers.

### 4. Add Other Providers
```python
# Use Anthropic Claude
llm = factory.create_llm_client(provider=LLMProvider.ANTHROPIC)

# Use OpenAI
llm = factory.create_llm_client(provider=LLMProvider.OPENAI)

# Back to Ollama
llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)
```

## API Endpoints

Your FastAPI server now has these AI endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ai/generate` | POST | Generate text from prompt |
| `/api/v1/ai/chat` | POST | Multi-turn conversation |
| `/api/v1/ai/providers` | GET | List available providers |
| `/api/v1/ai/health` | GET | Check AI services status |

## Example API Usage

### Generate Text
```bash
curl -X POST "http://localhost:8000/api/v1/ai/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain FastAPI in simple terms",
    "system_prompt": "You are a helpful teacher",
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

### Chat Conversation
```bash
curl -X POST "http://localhost:8000/api/v1/ai/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is AI?"},
      {"role": "assistant", "content": "AI is artificial intelligence"},
      {"role": "user", "content": "Give me an example"}
    ],
    "temperature": 0.7
  }'
```

### Check Available Providers
```bash
curl "http://localhost:8000/api/v1/ai/providers"
```

## Performance Tips

### Fast Responses
- Use `llama3.1:8b` (not 70b)
- Set `max_tokens=100` for short answers
- Enable GPU acceleration (automatic)

### Cost Optimization
- Ollama is 100% free (unlimited!)
- Only use OpenAI/Anthropic in production
- Cache common queries (coming soon)

### Production Deployment
- Use Docker for consistent environment
- Set `DEFAULT_LLM_PROVIDER=openai`
- Store API keys in secrets manager
- Monitor costs with built-in tracking

## Resources

- 📖 [Full AI Documentation](./README_AI.md)
- 🦙 [Ollama Models](https://ollama.com/library)
- 🐍 [FastAPI Docs](https://fastapi.tiangolo.com/)
- 💬 [Example Code](./examples/test_ollama.py)

## Success! 🎉

You now have:
- ✅ Free local LLM inference with Ollama
- ✅ Multi-provider abstraction (OpenAI, Anthropic, Ollama)
- ✅ FastAPI endpoints for AI generation
- ✅ Cost tracking and usage metrics
- ✅ Production-ready architecture

**Total Cost: $0.00** 🎊

Need help? Check [README_AI.md](./README_AI.md) for advanced features!