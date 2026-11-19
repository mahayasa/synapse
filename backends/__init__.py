"""Backend registry. Add a backend by adding one line here."""
from __future__ import annotations

import importlib
import os
import sys

# CTGAN's joblib process pool crashes on some Windows Python builds (resource tracker needs
# _posixsubprocess) once arrays are large. Run joblib sequentially there. Must be set before
# joblib is imported; override by setting JOBLIB_MULTIPROCESSING yourself.
if sys.platform == "win32":
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

from .base import Backend, BudgetExceeded, ConfigError  # noqa: F401

REGISTRY = {
    "rules": ("backends.rules_backend", "RulesBackend"),
    "anthropic_llm": ("backends.llm", "AnthropicBackend"),
    "openai_llm": ("backends.llm", "OpenAIBackend"),
    "openai_compat": ("backends.llm", "OpenAICompatBackend"),
    "sdv": ("backends.sdv_backend", "SDVBackend"),
}


def get_backend(cfg: dict) -> Backend:
    name = cfg.get("backend")
    if name not in REGISTRY:
        raise ConfigError(f"Unknown backend {name!r}. Choose one of: {', '.join(REGISTRY)}")
    module, cls = REGISTRY[name]
    return getattr(importlib.import_module(module), cls)(cfg)
