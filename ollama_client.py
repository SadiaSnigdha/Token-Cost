"""Small Ollama client and accounting helpers.

The client preserves Ollama's prompt_eval_count and eval_count exactly.
Those are template-inclusive counts, which is why the runner measures a
baseline call with the same system prompt.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen

INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60


class ChatBackend(Protocol):
    def chat(self, *, model: str, messages: list[dict[str, str]],
             options: dict[str, Any], keep_alive: int | str) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class CallResult:
    output: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Return the course-notional cost for Ollama token counts."""
    return (input_tokens * INPUT_PRICE_PER_MILLION
            + output_tokens * OUTPUT_PRICE_PER_MILLION) / 1_000_000


def result_from_response(response: dict[str, Any], latency_s: float) -> CallResult:
    """Convert one Ollama response into a validated, costed result."""
    try:
        message = response["message"]
        output = message["content"]
        input_tokens = response["prompt_eval_count"]
        output_tokens = response["eval_count"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Ollama response lacks message/count fields") from exc
    if not isinstance(output, str) or not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        raise ValueError("Ollama response contains invalid output or token types")
    return CallResult(output, input_tokens, output_tokens,
                      cost_usd(input_tokens, output_tokens), latency_s)


class OllamaHTTPBackend:
    """Minimal backend for Ollama's local /api/chat endpoint."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.url = base_url.rstrip("/") + "/api/chat"

    def chat(self, *, model: str, messages: list[dict[str, str]],
             options: dict[str, Any], keep_alive: int | str = 0) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": options,
        }
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed at {self.url}: {exc}") from exc


def run_call(backend: ChatBackend, *, model: str, system_prompt: str,
             prompt: str, temperature: float = 0.0, seed: int = 42,
             max_tokens: int = 128, keep_alive: int | str = 0) -> CallResult:
    """Run one independent chat call and return Ollama counts plus cost."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    started = time.perf_counter()
    response = backend.chat(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}],
        options={"temperature": temperature, "seed": seed, "num_predict": max_tokens},
        keep_alive=keep_alive,
    )
    return result_from_response(response, time.perf_counter() - started)
