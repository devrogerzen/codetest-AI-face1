"""Clase base abstracta: define la interfaz común y el manejo de errores/reintentos."""
import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider

logger = logging.getLogger("llm_client")


class BaseLLMClient(ABC):
    provider: Provider
    # Cada subclase define qué excepciones de su SDK se reintentan y cuáles no
    retryable_errors: tuple[type[Exception], ...] = ()
    fatal_errors: tuple[type[Exception], ...] = ()

    def __init__(self, config: ModelConfig):
        self.config = config

    # ---- Lo que implementa cada proveedor ----
    @abstractmethod
    async def _generate(self, messages: list[ChatMessage]) -> ModelResponse: ...

    @abstractmethod
    def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...

    # ---- API pública (igual para todos los proveedores) ----
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Respuesta completa. Nunca lanza: si falla, devuelve ok=False con el error."""
        for attempt in range(self.config.max_retries + 1):
            try:
                async with asyncio.timeout(self.config.timeout):
                    return await self._generate(messages)
            except self.fatal_errors as e:          # ej: API key inválida -> no reintentar
                return self._error(e)
            except (*self.retryable_errors, TimeoutError) as e:  # rate limit, red, timeout
                if attempt == self.config.max_retries:
                    return self._error(e)
                await self._backoff(attempt, e)
            except Exception as e:                  # cualquier otro error: controlado
                return self._error(e)
        return self._error(RuntimeError("Sin respuesta"))

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Generador asíncrono de tokens. Reintenta solo si todavía no se envió nada."""
        for attempt in range(self.config.max_retries + 1):
            sent = False
            try:
                async with asyncio.timeout(self.config.timeout):
                    async for token in self._stream(messages):
                        sent = True
                        yield token
                return
            except self.fatal_errors as e:
                yield f"\n[ERROR] {self._describe(e)}"
                return
            except (*self.retryable_errors, TimeoutError) as e:
                if sent or attempt == self.config.max_retries:
                    yield f"\n[ERROR] {self._describe(e)}"
                    return
                await self._backoff(attempt, e)
            except Exception as e:
                yield f"\n[ERROR] {self._describe(e)}"
                return

    # ---- Utilidades ----
    async def _backoff(self, attempt: int, e: Exception) -> None:
        wait = 2 ** attempt  # 1s, 2s, 4s... (backoff exponencial)
        logger.warning("%s: %s. Reintento %d en %ss", self.provider.value,
                       type(e).__name__, attempt + 1, wait)
        await asyncio.sleep(wait)  # await: no bloquea el event loop

    def _describe(self, e: Exception) -> str:
        return f"{type(e).__name__}: {str(e)[:200] or 'sin detalle'}"

    def _error(self, e: Exception) -> ModelResponse:
        logger.error("%s falló: %s", self.provider.value, self._describe(e))
        return ModelResponse(ok=False, provider=self.provider,
                             model=self.config.model, error=self._describe(e))
