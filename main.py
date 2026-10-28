"""Script de validación: prueba modo normal y streaming con cada proveedor configurado.

Uso:
    python main.py                 # usa LLM_PROVIDER del .env
    python main.py anthropic       # fuerza un proveedor
    python main.py all             # prueba todos los que tengan API key
"""
import asyncio
import logging
import os
import sys
import time

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client import AsyncLLMManager, LLMConfig, ModelConfig

PREGUNTA = "¿Qué es la entropía? Respondé en 2 oraciones."
SYSTEM = "Sos un profesor de física claro y conciso."


async def probar(provider: str) -> None:
    print(f"\n{'=' * 60}\nProveedor: {provider.upper()}\n{'=' * 60}")
    try:
        manager = AsyncLLMManager(LLMConfig.from_env(provider), temperature=0.3, max_tokens=200)
    except ValidationError as e:
        print(f"Configuración inválida: {e.errors()[0]['msg']}")
        return

    # 1) Modo normal
    print("\n[Modo normal]")
    t0 = time.perf_counter()
    resp = await manager.generate(PREGUNTA, system=SYSTEM)
    if resp.ok:
        print(resp.content)
        print(f"-> {resp.model} | tokens in/out: {resp.input_tokens}/{resp.output_tokens}"
              f" | {time.perf_counter() - t0:.2f}s")
    else:
        print(f"Error controlado: {resp.error}")

    # 2) Modo streaming (medimos TTFT)
    print("\n[Modo streaming]")
    t0, ttft = time.perf_counter(), None
    async for token in manager.stream(PREGUNTA, system=SYSTEM):
        if ttft is None and not token.startswith("\n[ERROR]"):
            ttft = time.perf_counter() - t0
        print(token, end="", flush=True)
    print(f"\n-> TTFT: {ttft:.2f}s | total: {time.perf_counter() - t0:.2f}s" if ttft else "")


def demo_validacion() -> None:
    """Muestra que Pydantic rechaza parámetros fuera de rango."""
    print("\n[Validación Pydantic] temperature=3.5 ->", end=" ")
    try:
        ModelConfig(model="x", temperature=3.5)
    except ValidationError as e:
        print(f"rechazado: {e.errors()[0]['msg']}")


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    arg = sys.argv[1] if len(sys.argv) > 1 else os.getenv("LLM_PROVIDER", "openai")

    demo_validacion()
    if arg == "all":
        providers = [p for p, k in (("openai", "OPENAI_API_KEY"),
                                    ("anthropic", "ANTHROPIC_API_KEY")) if os.getenv(k)]
        if not providers:
            print("No hay API keys configuradas en el .env")
        for p in providers:
            await probar(p)
    else:
        await probar(arg)


if __name__ == "__main__":
    asyncio.run(main())
