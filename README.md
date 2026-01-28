# 🤖 FastAPI AI Boilerplate

**Production-ready FastAPI boilerplate with multi-provider AI/LLM integration**

Build AI-powered applications with **zero cost during development** using local LLMs, then seamlessly switch to production providers (OpenAI, Anthropic) without code changes.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Wiki](https://img.shields.io/badge/docs-wiki-blue.svg)](https://github.com/ariefitriadin/fastapi-devbox-boilerplate/wiki)

---

## ✨ Key Features

### 🧠 **Multi-Provider LLM Integration**
- ✅ **Ollama** - FREE local inference (Llama3, Mistral, Gemma)
- ✅ **OpenAI** - GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- ✅ **Anthropic** - Claude 3.5 Sonnet, Opus, Haiku
- ✅ Runtime provider switching (no code changes)
- ✅ Automatic cost tracking & usage metrics

### 🚀 **Production Ready**
- ✅ FastAPI REST endpoints for AI generation
- ✅ Streaming responses for real-time UX
- ✅ Multi-turn conversations with context
- ✅ Type-safe configuration with Pydantic
- ✅ Comprehensive error handling
- ✅ Async/await throughout

### 💰 **Cost Optimized**
- ✅ $0.00 development costs with Ollama
- ✅ Per-request cost tracking
- ✅ Budget controls configured
- ✅ Smart provider routing patterns

### 🏗️ **Enterprise Architecture**
- ✅ PostgreSQL 17 with pgvector (ready for RAG)
- ✅ Redis caching layer (configured)
- ✅ Celery workers (ready for batch processing)
- ✅ Docker & Kubernetes ready
- ✅ Multi-cloud deployment (AWS/GCP/Azure)

---

## 🚀 Quick Start (5 Minutes)

### 1. Install Ollama (Free Local LLM)

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download

# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.1:8b

# Verify
ollama list
```

### 2. Setup Environment

```bash
# Enter devbox shell
devbox shell

# Copy environment template
cp env.ai.template .env

# .env should have:
# DEFAULT_LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.1:8b
```

### 3. Install Dependencies

```bash
poetry install
```

### 4. Start Database & API

```bash
# Initialize Postgres (first time only)
mkdir -p $PGHOST && mkdir -p $PGDATA
initdb -D $PGDATA

# Start Postgres
pg_ctl -D $PGDATA -o "-k $PGHOST -p 5433" start

# Run migrations
poetry run alembic upgrade head

# Start FastAPI server
poetry run uvicorn app.main:app --reload --port 8000
```

### 5. Test It!

```bash
# Option A: Run test suite
poetry run python examples/test_ollama.py

# Option B: Visit http://localhost:8000/docs
# Try POST /api/v1/ai/generate
```

**Total Development Cost: $0.00** ✨

---

## 💻 Usage Examples

### Simple Text Generation

```python
from app.services.ai import get_ai_factory
from app.core.ai_config import LLMProvider

# Create factory
factory = get_ai_factory()

# Use FREE Ollama for development
llm = factory.create_llm_client(provider=LLMProvider.OLLAMA)

# Generate text
response = await llm.generate("Explain quantum computing")
print(response.content)
print(f"Cost: ${response.usage.cost_usd:.4f}")  # $0.00!
print(f"Latency: {response.usage.latency_ms:.0f}ms")
```

### Switch to Production Provider

```python
# No code changes needed - just environment variable!
# .env: DEFAULT_LLM_PROVIDER=openai

llm = factory.create_llm_client()  # Uses OpenAI in production
response = await llm.generate("Complex reasoning task")
```

### REST API Example

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

**Response:**
```json
{
  "content": "FastAPI is a modern, high-performance web framework...",
  "model": "llama3.1:8b",
  "provider": "ollama",
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 42,
    "total_tokens": 57,
    "cost_usd": 0.0,
    "latency_ms": 823
  }
}
```

### Streaming Response

```python
async for token in llm.stream("Write a story about AI"):
    print(token, end="", flush=True)
```

### Multi-turn Chat

```python
from app.services.ai.llm.providers import LLMMessage

messages = [
    LLMMessage(role="system", content="You are a Python expert"),
    LLMMessage(role="user", content="What is FastAPI?"),
    LLMMessage(role="assistant", content="FastAPI is a web framework..."),
    LLMMessage(role="user", content="Show me an example")
]

response = await llm.chat(messages)
print(response.content)
```

---

## 🎯 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | FastAPI | High-performance async web framework |
| **Runtime** | Python 3.13 | Modern Python with performance improvements |
| **Database** | PostgreSQL 17 + pgvector | Relational DB with vector search |
| **ORM** | SQLAlchemy 2.0 (async) | Database abstraction layer |
| **AI/LLM** | Ollama, OpenAI, Anthropic | Multi-provider LLM integration |
| **Caching** | Redis | Response caching, semantic cache |
| **Workers** | Celery | Background job processing |
| **Environment** | Devbox (Nix) | Reproducible dev environment |
| **Deployment** | Docker, Kubernetes | Container orchestration |
| **Migrations** | Alembic | Database version control |

---

## 📁 Project Structure

```
apiboilerplate/
├── app/
│   ├── api/v1/endpoints/
│   │   └── ai.py              # AI REST endpoints
│   ├── core/
│   │   ├── config.py          # Base configuration
│   │   ├── ai_config.py       # AI-specific settings
│   │   └── database.py        # Database connection
│   ├── models/                # SQLAlchemy models
│   ├── schemas/               # Pydantic schemas
│   └── services/
│       └── ai/
│           ├── factory.py     # Provider factory (DI)
│           └── llm/providers/
│               ├── base.py              # Abstract base class
│               ├── ollama_provider.py  # FREE local inference
│               ├── openai_provider.py  # GPT-4, GPT-4o-mini
│               └── anthropic_provider.py # Claude 3.5
├── examples/
│   └── test_ollama.py         # Comprehensive test suite
├── data/
│   ├── models/yolo/           # Vision models (ready)
│   └── uploads/               # File uploads
├── migrations/                # Alembic migrations
├── QUICKSTART.md              # 5-minute setup guide
├── README_FASTAPI_CORE.md     # Base FastAPI documentation
└── docs/
    ├── SETUP_CHECKLIST.md     # Step-by-step checklist
    └── IMPLEMENTATION.md      # Technical architecture
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ai/generate` | POST | Generate text from prompt |
| `/api/v1/ai/chat` | POST | Multi-turn conversation |
| `/api/v1/ai/providers` | GET | List available providers |
| `/api/v1/ai/health` | GET | Check AI services status |
| `/health` | GET | System health check |

**Full API Documentation:** http://localhost:8000/docs (Swagger UI)

---

## 🌍 Supported AI Providers

### 🦙 Ollama (Recommended for Development)

**Cost:** FREE | **Speed:** Fast (local) | **Privacy:** 100% | **Limits:** None

```bash
# Setup
ollama pull llama3.1:8b

# .env
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

**Popular Models:**
- `llama3.1:8b` - Best general purpose (4.7GB)
- `mistral:7b` - Faster, efficient (4.1GB)
- `gemma2:9b` - Google's model (5.4GB)
- `codellama:7b` - Code generation (3.8GB)

### 🤖 OpenAI (Best for Production)

**Cost:** $0.15-$15/1M tokens | **Speed:** Very fast | **Quality:** Excellent

```bash
# Get API key: https://platform.openai.com/api-keys

# .env
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini  # Best value: $0.15/1M input
```

**Models:**
- `gpt-4o-mini` - **Best value** ($0.15/1M → $0.60/1M)
- `gpt-4o` - Balanced ($5/1M → $15/1M)
- `gpt-4-turbo` - Most capable ($10/1M → $30/1M)

### 🧠 Anthropic Claude

**Cost:** $0.25-$75/1M tokens | **Context:** 200K tokens | **Reasoning:** Superior

```bash
# Get API key: https://console.anthropic.com/

# .env
DEFAULT_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
```

**Models:**
- `claude-3-5-sonnet` - **Recommended** ($3/1M → $15/1M)
- `claude-3-opus` - Most powerful ($15/1M → $75/1M)
- `claude-3-haiku` - Fastest/cheapest ($0.25/1M → $1.25/1M)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[QUICKSTART.md](./QUICKSTART.md)** | Get started in 5 minutes |
| **[docs/SETUP_CHECKLIST.md](./docs/SETUP_CHECKLIST.md)** | Step-by-step setup checklist |
| **[docs/IMPLEMENTATION.md](./docs/IMPLEMENTATION.md)** | Architecture & design patterns |
| **[README_FASTAPI_CORE.md](./README_FASTAPI_CORE.md)** | Base FastAPI/Postgres setup |
| **[examples/test_ollama.py](./examples/test_ollama.py)** | Working code examples |
| **[env.ai.template](./env.ai.template)** | Environment configuration |

---

## 🚢 Deployment

### Local Development (FREE)

```bash
# Use Ollama - zero cost!
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

### Production (Docker)

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

COPY app/ ./app/
COPY migrations/ ./migrations/

ENV DEFAULT_LLM_PROVIDER=openai
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t fastapi-ai .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e DATABASE_URL=$DATABASE_URL \
  fastapi-ai
```

### Cloud Platforms

#### AWS
- **Compute:** ECS/Fargate or Lambda
- **Database:** RDS Postgres with pgvector
- **Cache:** ElastiCache Redis
- **Storage:** S3 for models/uploads

#### GCP
- **Compute:** Cloud Run or GKE
- **Database:** Cloud SQL Postgres
- **Cache:** Memorystore Redis
- **Storage:** Cloud Storage

#### Azure
- **Compute:** Container Apps or AKS
- **Database:** Azure Database for PostgreSQL
- **Cache:** Azure Cache for Redis
- **Storage:** Blob Storage

---

## 💰 Cost Optimization Strategies

### Strategy 1: Use Ollama for Development

```bash
# .env for local development
DEFAULT_LLM_PROVIDER=ollama  # FREE! Unlimited usage
```

**Savings:** 100% (vs. $0.10-$50/day with paid APIs)

### Strategy 2: Smart Provider Routing

```python
def get_llm_for_task(complexity: str):
    """Route to cheapest provider that meets requirements"""
    if complexity == "simple":
        # Free Ollama for simple tasks
        return factory.create_llm_client(provider=LLMProvider.OLLAMA)
    elif complexity == "medium":
        # gpt-4o-mini for balance
        return factory.create_llm_client(provider=LLMProvider.OPENAI)
    else:
        # Claude for complex reasoning
        return factory.create_llm_client(provider=LLMProvider.ANTHROPIC)
```

### Strategy 3: Monitor Costs

```python
# Per-request tracking
response = await llm.generate(prompt)
print(f"Request cost: ${response.usage.cost_usd:.4f}")

# Cumulative tracking
print(f"Total cost today: ${llm.get_total_cost():.4f}")
print(f"Total requests: {llm.get_request_count()}")

# Set budget alerts (coming soon)
if llm.get_total_cost() > DAILY_BUDGET:
    alert_admin()
```

### Strategy 4: Token Limits

```python
# Prevent runaway costs
response = await llm.generate(
    prompt,
    max_tokens=100,  # Limit output length
)
```

---

## 🛠️ Development Roadmap

### Phase 1: Core AI Integration ✅ COMPLETE

- [x] Multi-provider LLM abstraction
- [x] Ollama provider (FREE local)
- [x] OpenAI provider (GPT-4, GPT-4o-mini)
- [x] Anthropic provider (Claude 3.5)
- [x] REST API endpoints
- [x] Streaming support
- [x] Cost tracking
- [x] Comprehensive documentation

### Phase 2: Ready to Implement 🔜

- [ ] **Embeddings & RAG** (Vector search, semantic search)
- [ ] **YOLO Vision** (Object detection, image analysis)
- [ ] **Semantic Caching** (Reduce API calls by 80%+)
- [ ] **Background Workers** (Celery tasks for batch processing)
- [ ] **Observability** (LangSmith tracing, Prometheus metrics)

### Phase 3: Future Enhancements 💡

- [ ] Azure OpenAI provider
- [ ] Google Gemini integration
- [ ] Prompt template library
- [ ] A/B testing framework
- [ ] Cost optimization engine
- [ ] Multi-modal support (image + text)

---

## 🔧 Environment Configuration

### Minimal Setup (Ollama)

```bash
# .env
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
DATABASE_URL=postgresql+asyncpg://user@:5433/db
```

### Production Setup (OpenAI)

```bash
# .env
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.7

# Database
DATABASE_URL=postgresql+asyncpg://...

# Caching
REDIS_URL=redis://localhost:6379/0

# Observability
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
```

See [env.ai.template](./env.ai.template) for complete configuration options.

---

## 📜 Common Commands

| Task | Command |
|------|---------|
| **Enter devbox** | `devbox shell` |
| **Install deps** | `poetry install` |
| **Start Postgres** | `pg_ctl -D $PGDATA -o "-k $PGHOST -p 5433" start` |
| **Run migrations** | `poetry run alembic upgrade head` |
| **Start API** | `poetry run uvicorn app.main:app --reload` |
| **Run tests** | `poetry run python examples/test_ollama.py` |
| **Check DB status** | `pg_ctl -D $PGDATA status` |
| **Pull Ollama model** | `ollama pull llama3.1:8b` |
| **List models** | `ollama list` |

---

## ⚠️ Troubleshooting

### "Cannot connect to Ollama"

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Verify model is pulled
ollama list
```

### "OPENAI_API_KEY not found"

```bash
# Add to .env
echo "OPENAI_API_KEY=sk-proj-..." >> .env

# Or export temporarily
export OPENAI_API_KEY="sk-proj-..."
```

### Slow responses with Ollama

- Use smaller model: `mistral:7b` instead of `llama3.1:70b`
- Reduce `max_tokens` in requests
- Close other applications to free RAM
- Consider GPU acceleration (automatic if available)

### Database connection issues

```bash
# Check Postgres is running
pg_ctl -D $PGDATA status

# Verify DATABASE_URL in .env includes socket path
# Example: postgresql+asyncpg://user@:5433/db?host=/path/to/.devbox/virtenv/postgresql
```

---

## 🤝 Contributing

This is a specialized AI boilerplate maintained in the `project-ai` branch.

**Ways to contribute:**
- Report bugs or issues
- Suggest new AI providers
- Add example use cases
- Improve documentation
- Implement Phase 2 features

---

## 📄 License

See [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Ollama](https://ollama.com/) - Local LLM inference
- [OpenAI](https://openai.com/) - GPT models
- [Anthropic](https://anthropic.com/) - Claude models
- [Devbox](https://www.jetpack.io/devbox/) - Development environment

---

## 🎉 Ready to Build AI Applications!

You now have a production-ready AI boilerplate with:

✅ **Zero-cost development** (Ollama)  
✅ **Production providers** (OpenAI, Anthropic)  
✅ **REST API** ready to use  
✅ **Streaming support** for real-time UX  
✅ **Cost tracking** built-in  
✅ **Comprehensive docs** to guide you  

### Next Steps

1. **Get Started:** Follow [QUICKSTART.md](./QUICKSTART.md)
2. **Setup Guide:** Check [docs/SETUP_CHECKLIST.md](./docs/SETUP_CHECKLIST.md)
3. **Examples:** Run `python examples/test_ollama.py`
4. **API Docs:** Visit http://localhost:8000/docs
5. **Learn More:** Read [docs/IMPLEMENTATION.md](./docs/IMPLEMENTATION.md)

**Questions?** Check the documentation or open an issue.

**Happy coding!** 🚀
