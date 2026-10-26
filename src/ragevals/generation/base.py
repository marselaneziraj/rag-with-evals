"""LLM backend interface and token/cost accounting."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    backend: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_tokens(text: str) -> int:
    """~4 characters per token: close enough for cost reporting when the API
    does not return usage (Ollama does; the offline backend cannot)."""
    if not text:
        return 0
    words = len(re.findall(r"\S+", text))
    return max(1, int(round(0.75 * words + len(text) / 8)))


class LLMBackend:
    name = "base"
    supports_usage = False

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 700,
                 price_per_1k_input: float = 0.0, price_per_1k_output: float = 0.0,
                 **kwargs: Any) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.price_in = price_per_1k_input
        self.price_out = price_per_1k_output

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float | None = None) -> LLMResponse:
        raise NotImplementedError

    def price(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1000.0) * self.price_in + \
               (completion_tokens / 1000.0) * self.price_out

    def health(self) -> tuple[bool, str]:
        return True, "ok"

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model,
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "price_per_1k_input": self.price_in,
                "price_per_1k_output": self.price_out}
