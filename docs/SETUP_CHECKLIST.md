# ✅ AI Integration Setup Checklist

Quick reference guide to get your AI-powered FastAPI application running.

---

## 📋 Pre-requisites

- [ ] Python 3.13+ installed
- [ ] Devbox environment working
- [ ] Basic FastAPI boilerplate running (`/health` endpoint accessible)
- [ ] PostgreSQL database connected

---

## 🚀 Setup Steps

### Step 1: Install Ollama (Free Local LLM)

#### macOS / Linux
- [ ] Run: `curl -fsSL https://ollama.com/install.sh | sh`
- [ ] Verify: `ollama --version`

#### Windows
- [ ] Download from: https://ollama.com/download
- [ ] Install and verify: `ollama --version`

### Step 2: Start Ollama and Pull Model

- [ ] Start Ollama server: `ollama serve` (keep running in separate terminal)
- [ ] Pull Llama3.1 model: `ollama pull llama3.1:8b`
- [ ] Test model: `ollama run llama3.1:8b "Hello"`
- [ ] Verify models: `ollama list`

### Step 3: Configure Environment

- [ ] Copy template: `cp env.ai.template .env`
- [ ] Edit `.env` file
- [ ] Set: `DEFAULT_LLM_PROVIDER=ollama`
- [ ] Set: `OLLAMA_BASE_URL=http://localhost:11434`
- [ ] Set: `OLLAMA_MODEL=llama3.1:8b`

### Step 4: Install Python Dependencies

- [ ] Enter devbox shell: `devbox shell`
- [ ] Install packages: `poetry install`
- [ ] Wait for installation to complete (~5-10 minutes)

### Step 5: Test the Integration

#### Option A: Run Test Script
- [ ] Execute: `poetry run python examples/test_ollama.py`
- [ ] Verify all 7 tests pass
- [ ] Check that cost shows $0.00 (free!)

#### Option B: Start API Server
- [ ] Start server: `poetry run uvicorn app.main:app --reload --port 8000`
- [ ] Open browser: http://localhost:8000/docs
- [ ] Test endpoint: `POST /api/v1/ai/generate`
- [ ] Send test request:
  ```json
  {
    "prompt": "Explain Python in one sentence",
    "temperature": 0.7,
    "max_tokens": 100
  }
  ```
- [ ] Verify you get a response

### Step 6: Test API Endpoints

- [ ] Test generate: `POST /api/v1/ai/generate`
- [ ] Test chat: `POST /api/v1/ai/chat`
- [ ] Check providers: `GET /api/v1/ai/providers`
- [ ] Check health: `GET /api/v1/ai/health`

---

## 🔍 Verification Checklist

### Ollama Running
- [ ] `curl http://localhost:11434/api/tags` returns model list
- [ ] `ollama list` shows `llama3.1:8b`
- [ ] No errors in Ollama terminal window

### FastAPI Server
- [ ] Server starts without errors
- [ ] http://localhost:8000/docs loads
- [ ] Swagger UI shows `/api/v1/ai/*` endpoints
- [ ] `/health` endpoint returns `"status": "online"`

### AI Endpoints Working
- [ ] Generate endpoint returns text
- [ ] Response includes usage metrics
- [ ] Cost shows $0.00 with Ollama
- [ ] Latency is reasonable (<5 seconds)

### Files in Place
- [ ] `app/services/ai/` directory exists
- [ ] `app/services/ai/factory.py` exists
- [ ] `app/services/ai/llm/providers/ollama_provider.py` exists
- [ ] `app/api/v1/endpoints/ai.py` exists
- [ ] `.env` file configured

---

## 🐛 Troubleshooting

### "Cannot connect to Ollama"
- [ ] Check if Ollama is running: `curl http://localhost:11434/api/tags`
- [ ] If not running: `ollama serve`
- [ ] Check firewall settings
- [ ] Verify port 11434 is not blocked

### "Model not found"
- [ ] Run: `ollama list` to see available models
- [ ] Pull model: `ollama pull llama3.1:8b`
- [ ] Wait for download to complete
- [ ] Verify model appears in list

### "Import errors" in Python
- [ ] Ensure you're in devbox shell: `devbox shell`
- [ ] Reinstall dependencies: `poetry install`
- [ ] Check Python version: `python --version` (should be 3.13+)
- [ ] Clear cache: `poetry cache clear . --all`

### "Slow responses"
- [ ] Use smaller model: `ollama pull mistral:7b`
- [ ] Update `.env`: `OLLAMA_MODEL=mistral:7b`
- [ ] Reduce `max_tokens` in requests
- [ ] Close other applications to free memory

### "Out of memory"
- [ ] Use `llama3.1:8b` instead of `llama3.1:70b`
- [ ] Close other applications
- [ ] Check available RAM: should have 8GB+ free
- [ ] Consider using `mistral:7b` (smaller)

---

## 🎯 Next Steps (Optional)

### Add OpenAI Provider
- [ ] Get API key: https://platform.openai.com/api-keys
- [ ] Add to `.env`: `OPENAI_API_KEY=sk-proj-...`
- [ ] Add to `.env`: `OPENAI_MODEL=gpt-4o-mini`
- [ ] Test: Change `DEFAULT_LLM_PROVIDER=openai`
- [ ] Run test script to verify

### Add Anthropic Provider
- [ ] Get API key: https://console.anthropic.com/
- [ ] Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`
- [ ] Add to `.env`: `ANTHROPIC_MODEL=claude-3-5-sonnet-20240620`
- [ ] Test: Change `DEFAULT_LLM_PROVIDER=anthropic`
- [ ] Run test script to verify

### Enable Streaming
- [ ] Test streaming in examples: `python examples/test_ollama.py`
- [ ] Verify tokens appear in real-time
- [ ] Implement streaming in your endpoints

### Implement Caching (TODO)
- [ ] Install Redis: `brew install redis` or `apt install redis`
- [ ] Start Redis: `redis-server`
- [ ] Implement semantic cache (coming soon)

### Add YOLO Vision (TODO)
- [ ] Download YOLO model
- [ ] Implement vision service (coming soon)
- [ ] Test image detection

---

## 📚 Documentation Reference

- **Quick Start:** `QUICKSTART_AI.md` (5-minute setup)
- **Full Documentation:** `README_AI.md` (comprehensive guide)
- **Implementation Details:** `AI_IMPLEMENTATION_SUMMARY.md`
- **Environment Template:** `env.ai.template`
- **Test Examples:** `examples/test_ollama.py`

---

## ✅ Success Criteria

You're ready to go when:
- [x] Ollama is running and model is pulled
- [x] FastAPI server starts without errors
- [x] Test script passes all 7 tests
- [x] API endpoints return valid responses
- [x] Cost tracking shows $0.00 with Ollama
- [x] Documentation is accessible

---

## 🆘 Need Help?

1. **Check logs:** Look at FastAPI server output for errors
2. **Check Ollama:** Look at Ollama terminal for model loading issues
3. **Read docs:** `README_AI.md` has detailed troubleshooting
4. **Test basics:** Run `examples/test_ollama.py` to isolate issues
5. **Verify config:** Check `.env` file matches `env.ai.template`

---

## 🎉 You're Done!

If all checkboxes are checked, you now have:
- ✅ Free local LLM inference (Ollama)
- ✅ Multi-provider support (OpenAI, Anthropic, Ollama)
- ✅ REST API endpoints for AI
- ✅ Cost tracking and metrics
- ✅ Production-ready architecture
- ✅ Comprehensive documentation

**Total Development Cost: $0.00** 🎊

Ready to build AI-powered applications!