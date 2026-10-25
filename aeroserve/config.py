from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "AeroServe Gateway"
    ENVIRONMENT: str = "development"

    # Infrastructure
    REDIS_URL: str = "redis://redis:6379/0"
    DATABASE_URL: str = (
        "postgresql://aeroserve:aeroserve_password@postgres:5432/aeroserve_logs"
    )

    # Priority Queue (Platform-wide weights)
    BASE_WEIGHTS: dict[str, int] = Field(
        default_factory=lambda: {
            "premium": 5,
            "standard": 3,
            "background": 1,
        }
    )

    # Dynamic Batching Fallbacks
    DEFAULT_MAX_BATCH_SIZE: int = 16
    DEFAULT_BATCH_TIMEOUT_MS: int = 50

    # Circuit Breaker & Health Monitoring
    HEARTBEAT_INTERVAL_SEC: float = 1.0
    HEARTBEAT_TTL_SEC: int = 3
    CB_FAILURE_THRESHOLD: int = 2
    CB_CONSECUTIVE_SUCCESS_THRESHOLD: int = 3

    # Autoscaler Settings
    SCALE_UP_COOLDOWN_SEC: int = 30
    SCALE_DOWN_COOLDOWN_SEC: int = 120
    SCALE_DOWN_HYSTERESIS_FACTOR: float = 0.5
    MAX_WORKERS: int = 8
    MIN_WORKERS: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()