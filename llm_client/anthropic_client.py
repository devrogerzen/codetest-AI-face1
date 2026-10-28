"""Implementación para Anthropic usando el SDK asíncrono (AsyncAnthropic)."""
import inspect
from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic
from anthropic.resources.messages import AsyncMessages

from .base import BaseLLMClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider

# Las versiones 1.x del SDK quitaron 'temperature'; lo enviamos solo si existe
_SUPPORTS_TEMPERATURE = "temperature" in inspect.signature(AsyncMessages.create).parameters


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC
    retryable_errors = (anthropic.RateLimitError, anthropic.APIConnectionError,
                        anthropic.APITimeoutError, anthropic.InternalServerError)
    fatal_errors = (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                    anthropic.BadRequestError, anthropic.NotFoundError)

    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=api_key, max_retries=0)

    def _params(self, messages: list[ChatMessage]) -> dict:
        # Diferencia clave: Anthropic recibe el system prompt APARTE
        system = "\n".join(m.content for m in messages if m.role == "system")
        params = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages if m.role != "system"],
            "max_tokens": self.config.max_tokens,  # obligatorio en Anthropic
        }
        if _SUPPORTS_TEMPERATURE:
            params["temperature"] = min(self.config.temperature, 1.0)  # Anthropic admite 0-1
        if system:
            params["system"] = system
        return params

    async def _generate(self, messages: list[ChatMessage]) -> ModelResponse:
        resp = await self.client.messages.create(**self._params(messages))
        text = "".join(b.text for b in resp.content if b.type == "text")  # <- forma de Anthropic
        return ModelResponse(
            content=text,
            provider=self.provider,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    async def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with self.client.messages.stream(**self._params(messages)) as stream:
            async for text in stream.text_stream:
                yield text
