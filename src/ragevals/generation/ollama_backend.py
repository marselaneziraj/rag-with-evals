"""Local models through Ollama (http://localhost:11434).

Ollama is the no-key path to a *real* LLM: ``ollama pull llama3.1:8b`` and set
``generation.backend: ollama``. Cost is reported as 0 USD, which is true for
marginal API spend and false for electricity and wall-clock -- p95 latency is
the column to watch when comparing a local model with an API model.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from ..logging_utils import get_logger
from .base import LLMBackend, LLMResponse, estimate_tokens

log = get_logger(__name__)


class OllamaBackend(LLMBackend):
    name = "ollama"
    supports_usage = True

    def __init__(self, model: str = "llama3.1:8b", temperature: float = 0.0,
                 max_tokens: int = 700, base_url: str = "http://localhost:11434",
                 timeout_s: float = 120.0, price_per_1k_input: float = 0.0,
                 price_per_1k_output: float = 0.0, **kwargs) -> None:
        super().__init__(model, temperature, max_tokens, price_per_1k_input,
                         price_per_1k_output)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float | None = None) -> LLMResponse:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens,
                "seed": 0,            # determinism, as far as the runtime allows
            },
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": prompt}],
        }
        try:
            data = self._post("/api/chat", payload)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Ollama request failed against {self.base_url}: {exc}. "
                f"Is 'ollama serve' running and has '{self.model}' been pulled?"
            ) from exc
        text = (data.get("message") or {}).get("content", "").strip()
        pt = int(data.get("prompt_eval_count") or estimate_tokens(prompt))
        ct = int(data.get("eval_count") or estimate_tokens(text))
        return LLMResponse(text=text, model=self.model, backend=self.name,
                           prompt_tokens=pt, completion_tokens=ct,
                           latency_s=time.perf_counter() - started,
                           cost_usd=self.price(pt, ct), raw={})

    def health(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
            names = [m.get("name", "") for m in tags.get("models", [])]
            if self.model not in names:
                return False, (f"model {self.model!r} not pulled; available: "
                               f"{', '.join(names) or 'none'}")
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            return False, f"cannot reach {self.base_url}: {exc}"
