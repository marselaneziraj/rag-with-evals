"""Backend selection, with a safe fallback so the repo never hard-fails."""
from __future__ import annotations

from ..config import Config
from ..logging_utils import get_logger
from .base import LLMBackend
from .offline import OfflineExtractiveBackend
from .ollama_backend import OllamaBackend
from .openai_backend import OpenAIBackend

log = get_logger(__name__)

BACKENDS = {"offline": OfflineExtractiveBackend, "ollama": OllamaBackend,
            "openai": OpenAIBackend}


def build_backend(cfg: Config, *, strict: bool = False) -> LLMBackend:
    gen = cfg.generation
    if gen.backend not in BACKENDS:
        raise ValueError(f"unknown generation backend {gen.backend!r}; "
                         f"choose from {sorted(BACKENDS)}")
    kwargs = dict(model=gen.model, temperature=gen.temperature,
                  max_tokens=gen.max_tokens,
                  price_per_1k_input=gen.price_per_1k_input,
                  price_per_1k_output=gen.price_per_1k_output,
                  base_url=gen.base_url, api_base=gen.api_base,
                  api_key_env=gen.api_key_env, timeout_s=gen.timeout_s,
                  simulate_injection_susceptibility=gen.simulate_injection_susceptibility,
                  coverage_threshold=gen.coverage_threshold,
                  max_sentences=gen.max_sentences)
    backend = BACKENDS[gen.backend](**kwargs)
    ok, message = backend.health()
    if ok:
        return backend
    if strict:
        raise RuntimeError(f"{gen.backend} backend unhealthy: {message}")
    log.warning("%s backend unavailable (%s); falling back to the offline backend. "
                "Results will be labelled accordingly.", gen.backend, message)
    return OfflineExtractiveBackend(
        simulate_injection_susceptibility=gen.simulate_injection_susceptibility,
        coverage_threshold=gen.coverage_threshold,
        max_sentences=gen.max_sentences)
