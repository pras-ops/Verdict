from __future__ import annotations

from .base import Backend, BackendError, LogprobsUnsupported
from .huggingface import HuggingFaceBackend
from .mock import MockBackend
from .ollama import OllamaBackend
from .openai_compat import OpenAICompatBackend

BACKENDS: dict[str, type[Backend]] = {
    b.kind: b for b in (OllamaBackend, OpenAICompatBackend, HuggingFaceBackend, MockBackend)
}


def make_backend(kind: str, **config) -> Backend:
    try:
        cls = BACKENDS[kind]
    except KeyError:
        raise ValueError(f"unknown backend '{kind}', choose from {sorted(BACKENDS)}") from None
    return cls(**config)


def register_backend(cls: type[Backend]) -> type[Backend]:
    """Decorator for plugging in your own connector."""
    BACKENDS[cls.kind] = cls
    return cls


__all__ = ["Backend", "BackendError", "LogprobsUnsupported", "BACKENDS", "make_backend", "register_backend"]
