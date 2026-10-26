"""OpenAI-compatible Chat Completions backend.

Works with api.openai.com and with any gateway that speaks the same schema
(``generation.api_base``). The key is read from the environment variable named
in ``generation.api_key_env`` -- never from the config file, never logged.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..cost import lookup_price
from ..logging_utils import get_logger
from .base import LLMBackend, LLMResponse, estimate_tokens

log = get_logger(__name__)


class OpenAIBackend(LLMBackend):
    name = "openai"
    supports_usage = True

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0,
                 max_tokens: int = 700, api_base: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", timeout_s: float = 120.0,
                 price_per_1k_input: float = 0.0, price_per_1k_output: float = 0.0,
                 **kwargs) -> None:
        if not price_per_1k_input and not price_per_1k_output:
            price_per_1k_input, price_per_1k_output = lookup_price(model)
        super().__init__(model, temperature, max_tokens, price_per_1k_input,
                         price_per_1k_output)
        self.api_base = api_base.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"environment variable {self.api_key_env} is not set. Either export "
                f"it, or switch to a key-free backend: generation.backend=ollama "
                f"(local model) or generation.backend=offline (deterministic stub).")
        return key

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float | None = None) -> LLMResponse:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "seed": 0,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach {self.api_base}: {exc.reason}") from exc

        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or estimate_tokens(prompt))
        ct = int(usage.get("completion_tokens") or estimate_tokens(text))
        return LLMResponse(text=text, model=self.model, backend=self.name,
                           prompt_tokens=pt, completion_tokens=ct,
                           latency_s=time.perf_counter() - started,
                           cost_usd=self.price(pt, ct),
                           raw={"finish_reason": data["choices"][0].get("finish_reason")})

    def health(self) -> tuple[bool, str]:
        try:
            _ = self.api_key
        except RuntimeError as exc:
            return False, str(exc)
        return True, "ok"
