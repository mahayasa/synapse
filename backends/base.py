"""Common interface for every generation backend."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ConfigError(Exception):
    """Bad or missing configuration (never includes secret values)."""


class BudgetExceeded(Exception):
    """The configured token/call budget would be exceeded."""


class Backend(ABC):
    """A backend turns a config into CSV files in `out_dir`.

    generate() returns a dict of run metadata that is written to run_log.json.
    It must never put API keys in that dict.
    """

    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def generate(self, out_dir: Path) -> dict:
        ...

    def label(self) -> str:
        """Short label used in the output folder name."""
        return self.name
