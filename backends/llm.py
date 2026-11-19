"""LLM backends (Anthropic, OpenAI, any OpenAI-compatible endpoint).

Two modes:
  write_generator  the model writes a seeded Python script, we run it, validate,
                   and feed errors back for repair. Scales and is reproducible.
  direct_rows      the model emits CSV rows in chunks. Slow, costly, not
                   reproducible; fine for tiny tables and for benchmarking arm A.
"""
from __future__ import annotations

import io
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .base import Backend, BudgetExceeded, ConfigError
from .fewshot import build_few_shot
from .common import (extract_code, get_api_key, load_profile, run_script, run_validation)

GEN_SYSTEM = """You write Python scripts that generate synthetic relational datasets.
Requirements:
- Use only the standard library, numpy and pandas.
- CLI: --seed (int), --rows (int, rows of the main fact table), --out (output directory).
- Use ONE numpy.random.default_rng(seed); no global random state, no datetime.now().
- Write one CSV per table into --out. Generate dimension tables first, facts second.
- Primary keys unique, zero orphan foreign keys, derive dependent columns (total = qty * price).
- Model realism: seasonality, skewed (Zipf/lognormal) distributions, correlated columns.
- Inject a small share of anomalies and mark them with a boolean is_anomaly column.
- Use fake identifiers only; emails end with @example.com.
- If a "Real-data reference" is given, match its distributions, ranges, category mixes and ID
  formats, but never copy example rows or reproduce real identifiers.
Reply with ONE fenced python code block and nothing else."""

ROWS_SYSTEM = """You generate synthetic CSV data. Reply with ONLY the CSV (header line first),
no commentary, no code fences. Keys must be unique and consistent with the context given."""


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""


class AnthropicClient(LLMClient):
    def __init__(self, cfg: dict):
        try:
            import anthropic
        except ImportError as e:
            raise ConfigError("pip install anthropic") from e
        self.client = anthropic.Anthropic(api_key=get_api_key(cfg))
        self.model = cfg["model"]
        self.max_tokens = int(cfg.get("max_tokens", 8000))
        self.temperature = cfg.get("temperature")

    def complete(self, system, user):
        kw = {} if self.temperature is None else {"temperature": self.temperature}
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                        messages=[{"role": "user", "content": user}], **kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens


class OpenAICompatClient(LLMClient):
    """OpenAI, or any server speaking the OpenAI chat API (Ollama, vLLM, OpenRouter, ...)."""

    def __init__(self, cfg: dict):
        try:
            import openai
        except ImportError as e:
            raise ConfigError("pip install openai") from e
        # Local servers may need no key; a hosted one must name its variable.
        key = get_api_key(cfg, required=not cfg.get("no_api_key", False)) or "not-needed"
        self.client = openai.OpenAI(api_key=key, base_url=cfg.get("base_url"))
        self.model = cfg["model"]
        self.max_tokens = int(cfg.get("max_tokens", 8000))
        # Newer OpenAI models want max_completion_tokens; many servers want max_tokens.
        self.tokens_param = cfg.get("max_tokens_param", "max_tokens")
        self.temperature = cfg.get("temperature")

    def complete(self, system, user):
        kw = {self.tokens_param: self.max_tokens}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        r = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": user}], **kw)
        u = r.usage
        return r.choices[0].message.content or "", getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)


class LLMBackend(Backend):
    name = "llm"

    def __init__(self, cfg: dict, client: LLMClient | None = None):
        super().__init__(cfg)
        if not cfg.get("model"):
            raise ConfigError("`model` is required for LLM backends")
        self.few_shot = build_few_shot(cfg)      # raises early on unsafe/bad few_shot config
        self.client = client or self.make_client(cfg)
        self.tokens_in = self.tokens_out = self.calls = 0
        self.max_calls = int(cfg.get("max_calls", 12))
        self.max_total_tokens = int(cfg.get("max_total_tokens", 400_000))

    def make_client(self, cfg) -> LLMClient:  # overridden
        raise NotImplementedError

    def label(self):
        return f"{self.name}_{re.sub(r'[^A-Za-z0-9._-]', '-', self.cfg['model'])}"

    # -- budget-guarded call ---------------------------------------------------
    def _ask(self, system: str, user: str) -> str:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"max_calls={self.max_calls} reached")
        if self.tokens_in + self.tokens_out >= self.max_total_tokens:
            raise BudgetExceeded(f"max_total_tokens={self.max_total_tokens} reached")
        text, ti, to = self.client.complete(system, user)
        self.calls += 1
        self.tokens_in += ti
        self.tokens_out += to
        return text

    # -- modes ---------------------------------------------------------------------
    def generate(self, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = self.cfg.get("mode", "write_generator")
        t0 = time.time()
        if mode == "write_generator":
            extra = self._write_generator(out_dir)
        elif mode == "direct_rows":
            extra = self._direct_rows(out_dir)
        else:
            raise ConfigError(f"unknown mode {mode!r} (write_generator | direct_rows)")
        return {"model": self.cfg["model"], "mode": mode, "llm_calls": self.calls,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "llm_seconds": round(time.time() - t0, 1), **extra}

    def _task_text(self) -> str:
        c = self.cfg
        parts = [f"Domain: {c.get('domain', 'unspecified')}.",
                 f"Main fact table rows: {c.get('rows', 10000)}. Date range: "
                 f"{c.get('start_date', '12 months ending 2025-12-31')}.",
                 c.get("description", "")]
        profile = load_profile(c.get("domain_profile"))
        if profile:
            parts.append("Domain profile to follow:\n" + profile)
        if self.few_shot:
            parts.append(self.few_shot)
        return "\n".join(p for p in parts if p)

    def _write_generator(self, out_dir: Path) -> dict:
        script = out_dir / "generator.py"
        seed, rows = self.cfg.get("seed", 42), self.cfg.get("rows", 10_000)
        user = self._task_text()
        history, ok, log = [], False, ""
        for attempt in range(int(self.cfg.get("max_repairs", 3)) + 1):
            text = self._ask(GEN_SYSTEM, user)
            code = extract_code(text)
            script.write_text(code, encoding="utf-8")
            ok, log = run_script(script, ["--seed", str(seed), "--rows", str(rows),
                                          "--out", str(out_dir)], cwd=out_dir,
                                 timeout=int(self.cfg.get("script_timeout", 300)))
            if ok:
                ok, log = run_validation(out_dir, self.cfg.get("validation"))
            history.append({"attempt": attempt + 1, "ok": ok})
            if ok:
                break
            user = (self._task_text() + "\n\nYour previous script:\n```python\n" + code +
                    "\n```\nfailed with:\n" + log + "\nFix it and return the full corrected script.")
        if not ok:
            raise RuntimeError(f"no valid generator after {len(history)} attempts:\n{log}")
        return {"attempts": history, "generator_script": str(script)}

    def _direct_rows(self, out_dir: Path) -> dict:
        tables = self.cfg.get("tables")
        if not tables:
            raise ConfigError("direct_rows needs a `tables:` list (name, rows, columns)")
        chunk = int(self.cfg.get("rows_per_call", 40))
        made: dict[str, pd.DataFrame] = {}
        for t in tables:
            frames, have = [], 0
            while have < t["rows"]:
                ctx = [f"Table {n}: key column {df.columns[0]}, sample keys: "
                       f"{', '.join(map(str, df.iloc[:30, 0]))}" for n, df in made.items()]
                last = f"Last key so far: {frames[-1].iloc[-1, 0]}. Continue after it." if frames else ""
                want = min(chunk, t["rows"] - have)
                prompt = (f"{self._task_text()}\n\nGenerate {want} rows for table `{t['name']}` with "
                          f"columns: {t['columns']}.\n" + "\n".join(ctx) + f"\n{last}")
                text = self._ask(ROWS_SYSTEM, prompt)
                text = re.sub(r"^```(?:csv)?\s*|```\s*$", "", text.strip(), flags=re.M)
                try:
                    df = pd.read_csv(io.StringIO(text))
                except Exception as e:  # malformed CSV: skip this chunk, budget guard stops loops
                    continue
                frames.append(df)
                have += len(df)
            made[t["name"]] = pd.concat(frames, ignore_index=True).head(t["rows"])
            made[t["name"]].to_csv(out_dir / f"{t['name']}.csv", index=False)
        return {"tables": {n: len(d) for n, d in made.items()}, "reproducible": False}


class AnthropicBackend(LLMBackend):
    name = "anthropic_llm"

    def make_client(self, cfg):
        return AnthropicClient(cfg)


class OpenAICompatBackend(LLMBackend):
    name = "openai_compat"

    def make_client(self, cfg):
        return OpenAICompatClient(cfg)


class OpenAIBackend(OpenAICompatBackend):
    name = "openai_llm"
