"""Rule-based backend: runs a seeded generator script. No model, no API key."""
from __future__ import annotations

from pathlib import Path

from .base import Backend, ConfigError
from .common import SKILL_DIR, run_script


class RulesBackend(Backend):
    name = "rules"

    def generate(self, out_dir: Path) -> dict:
        gen = Path(self.cfg.get("generator", SKILL_DIR / "scripts" / "generator_template.py"))
        if not gen.is_absolute():
            gen = SKILL_DIR / gen
        if not gen.exists():
            raise ConfigError(f"generator script not found: {gen}")
        rows_arg = self.cfg.get("rows_arg", "--orders")
        out_dir.mkdir(parents=True, exist_ok=True)
        ok, log = run_script(gen, ["--seed", str(self.cfg.get("seed", 42)),
                                   rows_arg, str(self.cfg.get("rows", 10_000)),
                                   "--out", str(out_dir)], cwd=out_dir)
        if not ok:
            raise RuntimeError(f"generator failed:\n{log}")
        return {"generator": str(gen), "model": None}
