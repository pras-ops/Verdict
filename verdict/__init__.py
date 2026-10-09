"""Verdict: turn any LLM into a decision engine."""
from .backends import Backend, make_backend, register_backend
from .calibration import evaluate, fit_temperature
from .engine import Verdict
from .types import Decision, Result

__all__ = ["Verdict", "Decision", "Result", "Backend", "make_backend", "register_backend", "evaluate", "fit_temperature"]
__version__ = "0.1.0"
