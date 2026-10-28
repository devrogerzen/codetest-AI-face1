"""Esquemas Pydantic: validan todo lo que entra y sale del cliente."""
import os
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación (formato común a ambos proveedores)."""
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Parámetros del modelo, validados (ej: temperature fuera de 0-2 -> error)."""
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=8192)
    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)


class LLMConfig(BaseModel):
    """Configuración global. Las keys son SecretStr: nunca se imprimen en logs."""
    provider: Provider = Provider.OPENAI
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"

    @model_validator(mode="after")
    def check_key_for_provider(self) -> "LLMConfig":
        # Solo exige la key del proveedor que se va a usar
        if self.provider == Provider.OPENAI and not self.openai_api_key:
            raise ValueError("Falta OPENAI_API_KEY para el proveedor 'openai'")
        if self.provider == Provider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError("Falta ANTHROPIC_API_KEY para el proveedor 'anthropic'")
        return self

    @classmethod
    def from_env(cls, provider: str | None = None) -> "LLMConfig":
        """Arma la config desde variables de entorno (.env)."""
        data = {
            "provider": provider or os.getenv("LLM_PROVIDER", "openai"),
            "openai_api_key": os.getenv("OPENAI_API_KEY") or None,
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY") or None,
        }
        if os.getenv("OPENAI_MODEL"):
            data["openai_model"] = os.getenv("OPENAI_MODEL")
        if os.getenv("ANTHROPIC_MODEL"):
            data["anthropic_model"] = os.getenv("ANTHROPIC_MODEL")
        return cls(**data)


class ModelResponse(BaseModel):
    """Respuesta unificada: misma forma sin importar el proveedor."""
    ok: bool = True
    content: str = ""
    provider: Provider
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
