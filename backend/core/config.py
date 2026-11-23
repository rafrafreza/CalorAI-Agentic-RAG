"""Typed application settings (environment variables / ``.env``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

VERSION = "2.0.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str | None = Field(None, validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"))
    # "auto": Gemini when a key is configured, otherwise the deterministic offline agent.
    llm_mode: Literal["auto", "gemini", "offline"] = Field("auto", validation_alias="CALORAI_LLM_MODE")
    chat_model: str = Field("gemini-3.1-flash-lite", validation_alias="CALORAI_CHAT_MODEL")
    vision_model: str = Field("gemini-3.1-flash-lite", validation_alias="CALORAI_VISION_MODEL")
    memory_model: str = Field("gemini-3.1-flash-lite", validation_alias="CALORAI_MEMORY_MODEL")
    embedding_model: str = Field("gemini-embedding-001", validation_alias="CALORAI_EMBEDDING_MODEL")
    embedding_dims: int = Field(768, validation_alias="CALORAI_EMBEDDING_DIMS")
    enable_dense: bool = Field(True, validation_alias="CALORAI_ENABLE_DENSE")
    llm_timeout_s: float = Field(45.0, validation_alias="CALORAI_LLM_TIMEOUT_S")

    upload_dir: Path = Field(Path("uploads"), validation_alias="CALORAI_UPLOAD_DIR")
    cache_dir: Path = Field(Path(".cache"), validation_alias="CALORAI_CACHE_DIR")
    max_upload_mb: float = Field(8.0, validation_alias="CALORAI_MAX_UPLOAD_MB")
    max_message_chars: int = Field(2000, validation_alias="CALORAI_MAX_MESSAGE_CHARS")
    history_turns: int = Field(10, validation_alias="CALORAI_HISTORY_TURNS")
    cors_origins: str = Field(
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
        validation_alias="CALORAI_CORS_ORIGINS",
    )
    static_dir: Path = Field(Path("frontend/dist"), validation_alias="CALORAI_STATIC_DIR")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        if self.llm_mode == "offline":
            return False
        return bool(self.google_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
