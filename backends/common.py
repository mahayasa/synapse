"""Shared helpers: API-key lookup, script execution, validation."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from .base import ConfigError

SKILL_DIR = Path(__file__).resolve().parent.parent
SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def get_api_key(cfg: dict, required: bool = True) -> str | None:
    """Read the key from the environment variable NAMED in cfg['api_key_env'].

    The config holds the variable name only. If the value looks like a pasted
    key instead of a name we refuse, without echoing it.
    """
    var = cfg.get("api_key_env")
    if not var:
        if required:
            raise ConfigError("api_key_env is not set. Put the NAME of the environment "
                              "variable that holds your key, e.g. ANTHROPIC_API_KEY.")
        return None
    if not _VAR_NAME.match(var) or var.lower().startswith(("sk-", "sk_")) or len(var) > 64:
        raise ConfigError("api_key_env must be an environment variable NAME, not the key itself. "
                          "Remove the key from the config and rotate it if it was committed.")
    val = os.environ.get(var, "").strip()
    if not val:
        raise ConfigError(f"Environment variable {var} is not set or is empty. Set it in your "
                          f"shell or a git-ignored .env file.")
    return val


def scrubbed_env() -> dict:
    """Environment for running untrusted generated code: secrets removed."""
    return {k: v for k, v in os.environ.items()
            if not any(h in k.upper() for h in SECRET_HINTS)}


def extract_code(text: str) -> str:
    """Return the largest fenced code block, or the whole text if none."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return max(blocks, key=len).strip() if blocks else text.strip()


def run_script(script: Path, args: list[str], cwd: Path, timeout: int = 300) -> tuple[bool, str]:
    """Run a Python script with secrets stripped from its environment.

    This is NOT a real sandbox (no filesystem or network isolation). Treat
    LLM-written code as untrusted and prefer running in a container for
    anything beyond local experiments.
    """
    try:
        p = subprocess.run([sys.executable, str(script), *args], cwd=str(cwd),
                           env=scrubbed_env(), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    tail = (p.stderr or p.stdout)[-2000:]
    return p.returncode == 0, tail


def run_validation(out_dir: Path, vcfg: dict | None) -> tuple[bool, str]:
    """Run scripts/validate.py using the config's `validation` section."""
    if not vcfg:
        return True, "skipped (no validation section in config)"
    args = ["--dir", str(out_dir)]
    for flag, key in (("--pk", "pk"), ("--fk", "fk"), ("--range", "range"), ("--notnull", "notnull")):
        if vcfg.get(key):
            args += [flag, *map(str, vcfg[key])]
    p = subprocess.run([sys.executable, str(SKILL_DIR / "scripts" / "validate.py"), *args],
                       capture_output=True, text=True)
    return p.returncode == 0, p.stdout[-4000:] + p.stderr[-1000:]


def load_profile(name: str | None) -> str:
    """Read references/<name>.md if present."""
    if not name:
        return ""
    p = SKILL_DIR / "references" / f"{name}.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""
