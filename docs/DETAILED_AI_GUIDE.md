# 🤖 AI Services Documentation

Complete guide for integrating LLM and AI capabilities into your FastAPI application.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Provider Configuration](#provider-configuration)
- [Usage Examples](#usage-examples)
- [Cost Optimization](#cost-optimization)
- [Architecture](#architecture)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This boilerplate includes a **multi-provider AI abstraction layer** that allows you to:

- ✅ **Switch between LLM providers** (OpenAI, Anthropic, Ollama) without code changes
- ✅ **Track costs and usage** automatically
- ✅ **Stream responses** in real-time
- ✅ **Local inference** with Ollama (100% free, private)
- ✅ **Production-ready** error handling and retries
- ✅ **Type-safe** with full IDE autocomplete support

### Supported Providers

| Provider | Models | Cost | Best For |
|----------|--------|------|----------|
| **Ollama** | Llama3.1, Mistral, Gemma | FREE | Local dev, privacy |
| **OpenAI** | GPT-4o, GPT-4o-mini | $0.15-$15/1M tokens | Production, quality |
| **Anthropic** | Claude 3.5 Sonnet | $3-$15/1M tokens | Reasoning, long context |
| **Azure OpenAI** | GPT-4 | Variable | Enterprise compliance |

---

## 🚀 Quick Start

### 1. Install Ollama (for free local development)

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from: https://ollama.com/download

# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.1:8b
```

### 2. Configure Environment

```bash
# Copy the template
cp env.ai.template .env

# Edit .env and set:
DEFAULT_LLM_PROVIDER=ollama  # Use Ollama for free local inference
```

### 3. Install Dependencies

```bash
# Enter devbox shell
devbox shell

# Install AI dependencies
poetry install
```

### 4. Test It!

```bash
# Run the example script
poetry run python examples/test_ollama.py
```

---

## 📦 Installation

### Core Dependencies

The AI services require these packages (already in `pyproject.toml`):

```toml
# LLM Providers
openai>=1.30.0          # OpenAI GPT models
anthropic>=0.25.0       # Claude models
ollama>=0.1.0           # Local Ollama models

# Utilities
httpx>=0.27.0           # Async HTTP client
tenacity>=8.3.0         # Retry logic
```

### Optional Dependencies

```bash
# For embeddings (RAG systems)
poetry add sentence-transformers

# For vision (YOLO)
poetry add ultralytics opencv-python-headless

# For caching
poetry add redis aiocache
```

---

## ⚙️ Provider Configuration

### Ollama (Recommended for Development)

**Pros:** Free, private, no API keys, unlimited usage  
**Cons:** Slower than cloud APIs, requires local resources

```bash
# .env
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

**Available models:**
- `llama3.1:8b` - Best general purpose (4.7GB)
- `llama3.1:70b` - Most capable (40GB, needs GPU)
- `mistral:7b` - Fast and efficient (4.1GB)
- `gemma2:9b` - Google's model (5.4GB)

```bash
# List all models
ollama list

# Pull a new model
ollama pull mistral:7b
```

### OpenAI (Best for Production)

**Pros:** Highest quality, fast, reliable  
**Cons:** Costs money, requires API key

```bash
# .env
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...  # Get from https://platform.openai.com/api-keys
OPENAI_MODEL=gpt-4o-mini    # Most cost-effective
```

**Model recommendations:**
- `gpt-4o-mini` - **Best value** ($0.15/1M input, $0.60/1M output)
- `gpt-4o` - Balanced performance ($5/1M input, $15/1M output)
- `gpt-4-turbo` - Maximum capability ($10/1M input, $30/1M output)

### Anthropic Claude

**Pros:** Best reasoning, 200K context window, ethical AI  
**Cons:** Costs money, slightly slower

```bash
# .env
DEFAULT_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...  # Get from https://console.anthropic.com/
ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
```

**Model options:**
- `claude-3-5-sonnet` - **Recommended** ($3/1M input, $15/1M output)
- `claude-3-opus` - Most powerful ($15/1M input, $75/1M output)
- `claude-3-haiku` - Fastest/cheapest ($0.25/1M input, $1.25/1M output)

---

## 💻 Usage Examples

### Basic Text Generation

```python
from app.services.ai import get_ai_factory
from app.core.ai_config import LLMProvider

# Create factory
factory = get_ai_factory()

# Get LLM client (uses default provider from .env)
llm = factory.create_llm_client()

# Generate text
response = await llm.generate("Explain quantum computing")
print(response.content)
print(f"Cost: ${response.usage.cost_usd:.4f}")
```

### Override Provider at Runtime

```python
# Use Ollama for free inference
llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)
response = await llm.generate("Write a poem")

# Use OpenAI for better quality
llm = factory.create_llm_client(provider=LLMProvider.OPENAI)
response = await llm.generate("Write a technical document")
```

### Multi-turn Conversation

```python
from app.services.ai.llm.providers import LLMMessage

messages = [
    LLMMessage(role="system", content="You are a Python expert"),
    LLMMessage(role="user", content="What is FastAPI?"),
    LLMMessage(role="assistant", content="FastAPI is a modern web framework..."),
    LLMMessage(role="user", content="Show me an example"),
]

response = await llm.chat(messages)
print(response.content)
```

### Streaming Responses

```python
async for token in llm.stream("Write a story about AI"):
    print(token, end="", flush=True)
```

### System Prompts & Parameters

```python
response = await llm.generate(
    prompt="Explain machine learning",
    system_prompt="You are a teacher. Explain concepts simply.",
    temperature=0.7,      # Creativity (0.0-2.0)
    max_tokens=500,       # Limit response length
)
```

### FastAPI Endpoint Example

```python
from fastapi import APIRouter, Depends
from app.services.ai import get_ai_factory, AIServiceFactory

router = APIRouter()

@router.post("/ai/generate")
async def generate_text(
    prompt: str,
    factory: AIServiceFactory = Depends(get_ai_factory)
):
    """Generate AI completion"""
    llm = factory.create_llm_client()
    response = await llm.generate(prompt)
    
    return {
        "content": response.content,
        "model": response.model,
        "provider": response.provider,
        "tokens": response.usage.total_tokens,
        "cost_usd": response.usage.cost_usd,
        "latency_ms": response.usage.latency_ms,
    }
```

### Cost Tracking

```python
# Make multiple requests
for question in questions:
    response = await llm.generate(question)
    print(f"Tokens: {response.usage.total_tokens}")

# Check cumulative stats
print(f"Total requests: {llm.get_request_count()}")
print(f"Total cost: ${llm.get_total_cost():.4f}")
```

---

## 💰 Cost Optimization

### Strategy 1: Use Ollama for Development

```python
# .env
DEFAULT_LLM_PROVIDER=ollama  # FREE!

# Switch to OpenAI only in production
if os.getenv("ENVIRONMENT") == "production":
    llm = factory.create_llm_client(provider=LLMProvider.OPENAI)
else:
    llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)
```

### Strategy 2: Smart Provider Routing

```python
def get_llm_for_task(task_complexity: str):
    """Route to cheapest provider that meets requirements"""
    if task_complexity == "simple":
        # Use Ollama for free
        return factory.create_llm_client(provider=LLMProvider.OLLAMA)
    elif task_complexity == "medium":
        # Use gpt-4o-mini ($0.15/1M)
        llm = factory.create_llm_client(provider=LLMProvider.OPENAI)
        return llm
    else:
        # Use Claude for complex reasoning
        return factory.create_llm_client(provider=LLMProvider.ANTHROPIC)
```

### Strategy 3: Token Limits

```python
# Limit max tokens to control costs
response = await llm.generate(
    prompt,
    max_tokens=100,  # Prevent runaway costs
)
```

### Strategy 4: Caching (TODO)

```python
# Semantic caching will avoid duplicate API calls
# Coming soon: app/services/ai/cache/semantic_cache.py
```

---

## 🏗️ Architecture

### Directory Structure

```
app/services/ai/
├── __init__.py
├── factory.py                          # Provider factory
├── llm/
│   ├── __init__.py
│   └── providers/
│       ├── __init__.py
│       ├── base.py                     # Abstract base class
│       ├── ollama_provider.py          # Ollama implementation
│       ├── openai_provider.py          # OpenAI implementation
│       └── anthropic_provider.py       # Anthropic implementation
└── (future: vision/, cache/, etc.)
```

### Design Patterns

#### 1. **Strategy Pattern** (Provider Abstraction)
All providers implement `BaseLLMProvider`, allowing runtime switching:

```python
# Change provider without code changes
llm = factory.create_llm_client(provider=LLMProvider.OPENAI)
# vs
llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)
# Same interface, different implementation
```

#### 2. **Factory Pattern** (Service Creation)
Centralized provider instantiation with caching:

```python
# Factory caches instances to avoid recreating clients
factory = AIServiceFactory(settings)
llm1 = factory.create_llm_client()  # Creates new
llm2 = factory.create_llm_client()  # Returns cached
```

#### 3. **Dependency Injection** (FastAPI Integration)
Use `Depends()` for clean endpoint code:

```python
@app.post("/generate")
async def generate(
    prompt: str,
    factory: AIServiceFactory = Depends(get_ai_factory)
):
    llm = factory.create_llm_client()
    return await llm.generate(prompt)
```

---

## 🚢 Deployment

### Local Development

```bash
# Use Ollama (free)
DEFAULT_LLM_PROVIDER=ollama

# Start services
devbox shell
ollama serve  # In separate terminal
poetry run uvicorn app.main:app --reload
```

### Production (Docker)

```dockerfile
# Dockerfile
FROM python:3.13-slim

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

# Copy application
COPY app/ ./app/

# Set production provider
ENV DEFAULT_LLM_PROVIDER=openai
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

**AWS:**
```bash
# Store API keys in AWS Secrets Manager
aws secretsmanager create-secret \
  --name openai-api-key \
  --secret-string "sk-proj-..."

# Reference in ECS task definition
"secrets": [
  {
    "name": "OPENAI_API_KEY",
    "valueFrom": "arn:aws:secretsmanager:..."
  }
]
```

**Environment Variables:**
```bash
# Required for production
OPENAI_API_KEY=sk-...
DEFAULT_LLM_PROVIDER=openai
ENVIRONMENT=production
```

---

## 🔧 Troubleshooting

### "Cannot connect to Ollama"

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running:
ollama serve

# Check if model is pulled
ollama list

# Pull model if missing
ollama pull llama3.1:8b
```

### "OPENAI_API_KEY not found"

```bash
# Add to .env file
echo "OPENAI_API_KEY=sk-proj-..." >> .env

# Or export temporarily
export OPENAI_API_KEY="sk-proj-..."
```

### "Model not found" Error

```bash
# Ollama: Pull the model
ollama pull llama3.1:8b

# OpenAI: Check model name spelling
# ✅ Correct: gpt-4o-mini
# ❌ Wrong: gpt-4-mini, gpt4o-mini
```

### Slow Response Times

**Ollama:**
- Use smaller models: `llama3.1:8b` instead of `llama3.1:70b`
- Use GPU if available: Models automatically use GPU when present
- Reduce max_tokens: `max_tokens=500` instead of unlimited

**OpenAI/Anthropic:**
- Check your internet connection
- Use streaming for perceived speed: `llm.stream(prompt)`
- Choose faster models: `gpt-4o-mini` over `gpt-4-turbo`

### High Costs

```python
# Monitor costs in code
print(f"Cost so far: ${llm.get_total_cost():.4f}")

# Set budget limits (TODO: implement in rate_limiter.py)
if llm.get_total_cost() > 10.0:
    raise RuntimeError("Daily budget exceeded")

# Use cheaper providers
llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)
```

---

## 📚 Next Steps

### Implement Embeddings & RAG
```python
# TODO: Coming soon
from app.services.ai.embeddings import create_embedding_client

embeddings = create_embedding_client(provider="local")
vectors = await embeddings.embed_documents(["doc1", "doc2"])
```

### Add YOLO Vision
```python
# TODO: Coming soon
from app.services.ai.vision import YOLOService

yolo = YOLOService(settings)
detections = await yolo.detect(image_path)
```

### Implement Caching
```python
# TODO: Semantic caching to reduce API calls
from app.services.ai.cache import SemanticCache

cache = SemanticCache(threshold=0.95)
response = await cache.get_or_generate(prompt, llm)
```

---

## 📖 Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 🤝 Contributing

Found a bug or want to add a new provider? Please open an issue or PR!

**Potential additions:**
- Azure OpenAI provider
- Google Gemini provider
- Cohere provider
- HuggingFace Inference API
- LangChain integration
- Prompt templates library

---

## 📄 License

See main project LICENSE file.