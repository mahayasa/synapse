"""Offline tests: fake LLM clients, no network, no API keys.  python -m unittest discover tests"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backends import BudgetExceeded, ConfigError, get_backend  # noqa: E402
from backends.common import extract_code, get_api_key, run_script, scrubbed_env  # noqa: E402
from backends.llm import AnthropicBackend, LLMClient  # noqa: E402
from run import run  # noqa: E402

GOOD = '''```python
import argparse
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int); ap.add_argument("--rows", type=int)
ap.add_argument("--out"); a = ap.parse_args()
rng = np.random.default_rng(a.seed)
c = pd.DataFrame({"customer_id": range(1, 21)})
o = pd.DataFrame({"order_id": range(1, a.rows + 1), "customer_id": rng.integers(1, 21, a.rows)})
c.to_csv(Path(a.out) / "customers.csv", index=False); o.to_csv(Path(a.out) / "orders.csv", index=False)
```'''
BAD = "```python\nraise SystemExit('boom')\n```"
VALID = {"pk": ["customers=customer_id", "orders=order_id"],
         "fk": ["orders.customer_id=customers.customer_id"]}


class Fake(LLMClient):
    def __init__(self, replies):
        self.replies, self.n = replies, 0

    def complete(self, system, user):
        r = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        self.last_user = user
        return r, 100, 50


class Tests(unittest.TestCase):
    def test_key_missing_and_key_pasted_are_rejected(self):
        os.environ.pop("SYN_TEST_KEY", None)
        with self.assertRaises(ConfigError):
            get_api_key({"api_key_env": "SYN_TEST_KEY"})
        with self.assertRaises(ConfigError) as cm:
            get_api_key({"api_key_env": "sk-ant-abc123secret"})
        self.assertNotIn("abc123secret", str(cm.exception))
        os.environ["SYN_TEST_KEY"] = "v"
        self.assertEqual(get_api_key({"api_key_env": "SYN_TEST_KEY"}), "v")
        self.assertIsNone(get_api_key({}, required=False))

    def test_generated_code_runs_without_secrets(self):
        os.environ["MY_SECRET_API_KEY"] = "leak"
        self.assertNotIn("MY_SECRET_API_KEY", scrubbed_env())
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "s.py"
            s.write_text("import os;print('x' if 'MY_SECRET_API_KEY' in os.environ else 'clean')")
            ok, _ = run_script(s, [], Path(d))
            self.assertTrue(ok)

    def test_extract_code(self):
        self.assertEqual(extract_code("hi ```python\nx=1\n``` bye"), "x=1")

    def test_rules_backend_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            log = run({"backend": "rules", "rows": 300, "seed": 1, "rows_arg": "--orders",
                       "validation": {"pk": ["orders=order_id"],
                                      "fk": ["orders.customer_id=customers.customer_id"]}}, Path(d))
            self.assertTrue(log["validation_passed"])
            self.assertTrue((Path(d) / "run_log.json").exists())

    def test_relative_out_dir(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                log = run({"backend": "rules", "rows": 100, "rows_arg": "--orders",
                           "validation": {"pk": ["orders=order_id"]}}, Path("rel/out"))
                self.assertTrue(log["validation_passed"])
            finally:
                os.chdir(cwd)

    def test_llm_repair_loop_and_log_has_no_key(self):
        os.environ["SYN_TEST_KEY"] = "topsecret"
        fake = Fake([BAD, GOOD])
        cfg = {"backend": "anthropic_llm", "model": "m", "rows": 50, "seed": 3,
               "api_key_env": "SYN_TEST_KEY", "validation": VALID}
        with tempfile.TemporaryDirectory() as d:
            b = AnthropicBackend(cfg, client=fake)
            meta = b.generate(Path(d))
            self.assertEqual([a["ok"] for a in meta["attempts"]], [False, True])
            self.assertIn("boom", fake.last_user)            # error fed back for repair
            self.assertEqual(meta["llm_calls"], 2)
            self.assertNotIn("topsecret", json.dumps(meta))

    def test_budget_guard(self):
        cfg = {"backend": "anthropic_llm", "model": "m", "rows": 10, "max_calls": 1,
               "max_repairs": 5, "validation": VALID}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(BudgetExceeded):
                AnthropicBackend(cfg, client=Fake([BAD])).generate(Path(d))

    def test_direct_rows(self):
        csv_t = "t_id,v\n" + "\n".join(f"{i},{i * 2}" for i in range(1, 6))
        cfg = {"backend": "anthropic_llm", "model": "m", "mode": "direct_rows", "rows_per_call": 5,
               "tables": [{"name": "t", "rows": 5, "columns": "t_id, v"}]}
        with tempfile.TemporaryDirectory() as d:
            meta = AnthropicBackend(cfg, client=Fake([csv_t])).generate(Path(d))
            self.assertEqual(meta["tables"], {"t": 5})

    def test_unknown_backend_and_sdv_without_real_data(self):
        with self.assertRaises(ConfigError):
            get_backend({"backend": "nope"})
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ConfigError):
            get_backend({"backend": "sdv"}).generate(Path(d))


if __name__ == "__main__":
    unittest.main()
