"""Factory + AsyncLLMManager: elige el proveedor según la configuración."""
from collections.abc import AsyncIterator

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .schemas import ChatMessage, LLMConfig, ModelConfig, ModelResponse, Provider


class LLMFactory:
    @staticmethod
    def create_client(config: LLMConfig, **model_params) -> BaseLLMClient:
        if config.provider == Provider.OPENAI:
            mc = ModelConfig(model=config.openai_model, **model_params)
            return OpenAIClient(config.openai_api_key.get_secret_value(), mc)
        if config.provider == Provider.ANTHROPIC:
            mc = ModelConfig(model=config.anthropic_model, **model_params)
            return AnthropicClient(config.anthropic_api_key.get_secret_value(), mc)
        raise ValueError(f"Proveedor no soportado: {config.provider}")


class AsyncLLMManager:
    """Punto de entrada único para la app. La lógica de negocio solo usa esto."""

    def __init__(self, config: LLMConfig | None = None, **model_params):
        self.config = config or LLMConfig.from_env()
        self.client = LLMFactory.create_client(self.config, **model_params)

    @staticmethod
    def _to_messages(prompt: str | list[ChatMessage], system: str | None) -> list[ChatMessage]:
        if isinstance(prompt, str):
            msgs = [ChatMessage(role="user", content=prompt)]
        else:
            msgs = list(prompt)
        if system:
            msgs.insert(0, ChatMessage(role="system", content=system))
        return msgs

    async def generate(self, prompt: str | list[ChatMessage],
                       system: str | None = None) -> ModelResponse:
        return await self.client.generate(self._to_messages(prompt, system))

    async def stream(self, prompt: str | list[ChatMessage],
                     system: str | None = None) -> AsyncIterator[str]:
        async for token in self.client.stream(self._to_messages(prompt, system)):
            yield token
