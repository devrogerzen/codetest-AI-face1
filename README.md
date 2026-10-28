# Unified Async LLM Client

Pre-entrega 1 de AI Engineering (Coderhouse). Es un cliente asíncrono que usa **OpenAI** o **Anthropic** con la misma interfaz. Soporta streaming, valida con Pydantic y maneja los errores sin romper el programa.

## Estructura

```
llm_client/
├── schemas.py           # Pydantic: ChatMessage, ModelConfig, LLMConfig (SecretStr), ModelResponse
├── base.py              # BaseLLMClient (ABC): interfaz común + reintentos + timeouts
├── openai_client.py     # OpenAIClient (AsyncOpenAI)
├── anthropic_client.py  # AnthropicClient (AsyncAnthropic)
└── manager.py           # LLMFactory + AsyncLLMManager
main.py                  # Script de prueba: modo normal + streaming
```

## Cómo correrlo

```bash
# 0. Clonar el repo
git clone https://github.com/devrogerzen/codetest-AI-face1.git
cd codetest-AI-face1

# 1. Entorno virtual con Python 3.12
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Dependencias
pip install -r requirements.txt

# 3. Variables de entorno
copy .env.example .env          # Windows (cp en Linux/Mac) y completá tus keys

# 4. Ejecutar
python main.py                  # proveedor de LLM_PROVIDER
python main.py anthropic        # forzar un proveedor
python main.py all              # todos los que tengan API key
```

## Variables de entorno

| Variable | Requerida | Descripción |
|---|---|---|
| `LLM_PROVIDER` | No (default `openai`) | `openai` o `anthropic` |
| `OPENAI_API_KEY` | Si se usa OpenAI | API key de OpenAI |
| `ANTHROPIC_API_KEY` | Si se usa Anthropic | API key de Anthropic |
| `OPENAI_MODEL` | No | Default `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | No | Default `claude-haiku-4-5` |

## Uso en código

```python
from llm_client import AsyncLLMManager

manager = AsyncLLMManager(temperature=0.5, max_tokens=300)  # lee el .env

resp = await manager.generate("¿Qué es la entropía?")
print(resp.content if resp.ok else resp.error)

async for token in manager.stream("¿Qué es la entropía?"):
    print(token, end="", flush=True)
```

## Decisiones de diseño

- **Intercambiabilidad:** `LLMFactory` crea el cliente según `LLM_PROVIDER`. La app solo usa `AsyncLLMManager`, así que cambiar de proveedor no toca la lógica de negocio.
- **Asincronía:** se usan los SDKs async oficiales (`AsyncOpenAI`, `AsyncAnthropic`) y nunca llamadas bloqueantes (`await asyncio.sleep` en vez de `time.sleep`).
- **Streaming:** `stream()` es un generador asíncrono (`async for` + `yield`).
- **Validación:** Pydantic valida los roles de los mensajes, `temperature` (0 a 2), `max_tokens` (>0) y que exista la key del proveedor elegido. Las keys son `SecretStr` y no aparecen en logs.
- **Diferencias abstraídas:**
  - Dónde está el texto: `choices[0].message.content` en OpenAI y `content[0].text` en Anthropic.
  - El system prompt va aparte en Anthropic.
  - `max_tokens` es obligatorio en Anthropic.
- **Errores:**
  - Rate limit, fallos de red, timeouts y errores 5xx → reintento con backoff exponencial (1s, 2s, 4s).
  - API key inválida o request incorrecto → sin reintento.
  - En todos los casos se devuelve `ModelResponse(ok=False, error=...)`, o `[ERROR] ...` en streaming. Nunca un crash.
  - En streaming solo se reintenta si todavía no se envió ningún token.
