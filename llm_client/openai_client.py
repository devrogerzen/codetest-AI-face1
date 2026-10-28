"""Implementación para OpenAI usando el SDK asíncrono (AsyncOpenAI)."""
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from .base import BaseLLMClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI
    retryable_errors = (openai.RateLimitError, openai.APIConnectionError,
                        openai.APITimeoutError, openai.InternalServerError)
    fatal_errors = (openai.AuthenticationError, openai.PermissionDeniedError,
                    openai.BadRequestError, openai.NotFoundError)

    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(config)
        # max_retries=0: los reintentos los maneja nuestra clase base
        self.client = AsyncOpenAI(api_key=api_key, max_retries=0)

    def _params(self, messages: list[ChatMessage]) -> dict:
        # OpenAI acepta el system prompt dentro de la lista de mensajes
        return {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    async def _generate(self, messages: list[ChatMessage]) -> ModelResponse:
        resp = await self.client.chat.completions.create(**self._params(messages))
        return ModelResponse(
            content=resp.choices[0].message.content or "",   # <- forma de OpenAI
            provider=self.provider,
            model=resp.model,
            input_tokens=resp.usage.prompt_tokens if resp.usage else None,
            output_tokens=resp.usage.completion_tokens if resp.usage else None,
        )

    async def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            **self._params(messages), stream=True)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
