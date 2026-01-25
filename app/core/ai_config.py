"""
AI Configuration with Provider Abstraction
Supports: OpenAI, Anthropic, Ollama, Azure OpenAI
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure-openai"
    OLLAMA = "ollama"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers"""

    LOCAL = "local"
    OPENAI = "openai"
    COHERE = "cohere"


class VectorStore(str, Enum):
    """Supported vector databases"""

    PGVECTOR = "pgvector"
    CHROMADB = "chromadb"
    PINECONE = "pinecone"


class AISettings(BaseSettings):
    """AI/ML Configuration with multi-provider support"""

    # ============================================
    # LLM Configuration
    # ============================================
    default_llm_provider: LLMProvider = LLMProvider.OPENAI

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_max_tokens: int = 4096
    openai_temperature: float = 0.7

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20240620"

    # Azure OpenAI
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_deployment_name: Optional[str] = None

    # Ollama (Local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # ============================================
    # Embedding Configuration
    # ============================================
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ============================================
    # Vector Store
    # ============================================
    vector_store: VectorStore = VectorStore.PGVECTOR
    vector_index_name: str = "embeddings"
    similarity_threshold: float = 0.75

    # ============================================
    # YOLO Configuration
    # ============================================
    yolo_model_version: str = "yolov8n"
    yolo_model_path: str = "./data/models/yolo/yolov8n.pt"
    yolo_confidence_threshold: float = 0.5
    yolo_iou_threshold: float = 0.45
    yolo_device: Literal["cpu", "cuda", "mps"] = "cpu"

    # ============================================
    # Task Queue
    # ============================================
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_task_time_limit: int = 300  # seconds
    worker_concurrency: int = 4

    # ============================================
    # Caching
    # ============================================
    redis_url: str = "redis://localhost:6379/2"
    cache_ttl: int = 3600
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95

    # ============================================
    # Rate Limiting & Budget
    # ============================================
    rate_limit_enabled: bool = True
    rate_limit_per_user: int = 100
    daily_budget_usd: float = 50.0
    max_tokens_per_request: int = 8000

    # ============================================
    # Observability
    # ============================================
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "my-ai-project"
    langsmith_tracing: bool = False
    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    # ============================================
    # File Storage
    # ============================================
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = Field(
        default_factory=lambda: ["pdf", "txt", "docx", "png", "jpg", "jpeg"]
    )
    storage_backend: Literal["local", "s3", "gcs", "azure-blob"] = "local"

    @validator("openai_api_key")
    def validate_openai_key(cls, v, values):
        """Ensure OpenAI key exists if it's the default provider"""
        if values.get("default_llm_provider") == LLMProvider.OPENAI and not v:
            raise ValueError("OPENAI_API_KEY required when using OpenAI provider")
        return v

    @validator("yolo_device")
    def validate_yolo_device(cls, v):
        """Warn if CUDA requested but not available"""
        if v == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    print("⚠️  CUDA requested but not available, falling back to CPU")
                    return "cpu"
            except ImportError:
                return "cpu"
        return v

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


ai_settings = AISettings()
