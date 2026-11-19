"""Few-shot context tests (offline)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import ConfigError  # noqa: E402
from backends.fewshot import build_few_shot  # noqa: E402
from backends.llm import AnthropicBackend  # noqa: E402
from test_backends import GOOD, Fake  # noqa: E402


def make_real(d: Path):
    rng = np.random.default_rng(0)
    n = 200
    pd.DataFrame({"email": [f"alice.{i}@realcorp.com" for i in range(n)],
                  "segment": rng.choice(["retail", "member"], n, p=[.8, .2]),
                  "amount": rng.lognormal(3, .5, n).round(2),
                  "ssn_like": [f"{i:09d}" for i in range(n)]}).to_csv(d / "customers.csv", index=False)


class FewShotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        make_real(self.d)

    def tearDown(self):
        self.tmp.cleanup()

    def cfg(self, **fs):
        return {"few_shot": {"real_data": str(self.d), **fs}}

    def test_off_by_default(self):
        self.assertEqual(build_few_shot({}), "")

    def test_stats_mode_has_shape_but_no_raw_values(self):
        t = build_few_shot(self.cfg())
        self.assertIn("customers", t)
        self.assertIn("segment (category", t)
        self.assertIn("retail=", t)
        self.assertNotIn("alice.", t)               # identifier values are masked
        self.assertNotIn("realcorp", t)
        self.assertRegex(t, r"masked format a+\.9+@a+\.a+")

    def test_rows_mode_requires_acknowledgement_unless_local(self):
        with self.assertRaises(ConfigError):
            build_few_shot(self.cfg(mode="rows"))
        t = build_few_shot(self.cfg(mode="rows", acknowledge_external_data=True, rows_per_table=3))
        self.assertIn("Example rows", t)
        local = {**self.cfg(mode="rows"), "base_url": "http://localhost:11434/v1"}
        self.assertIn("Example rows", build_few_shot(local))
        remote = {**self.cfg(mode="rows"), "base_url": "https://api.example.com/v1"}
        with self.assertRaises(ConfigError):
            build_few_shot(remote)

    def test_exclude_columns_and_truncation(self):
        t = build_few_shot(self.cfg(mode="rows", acknowledge_external_data=True, exclude_columns=["email", "ssn_like"]))
        self.assertNotIn("email", t)
        self.assertNotIn("ssn_like", t)
        short = build_few_shot(self.cfg(max_chars=200))
        self.assertIn("[reference truncated]", short)
        self.assertLessEqual(len(short), 230)

    def test_bad_config(self):
        with self.assertRaises(ConfigError):
            build_few_shot({"few_shot": {"real_data": "/nope"}})
        with self.assertRaises(ConfigError):
            build_few_shot(self.cfg(mode="weird"))

    def test_reference_reaches_the_llm_prompt(self):
        fake = Fake([GOOD])
        cfg = {"backend": "anthropic_llm", "model": "m", "rows": 20, "seed": 1,
               "few_shot": {"real_data": str(self.d)},
               "validation": {"pk": ["orders=order_id"]}}
        with tempfile.TemporaryDirectory() as out:
            AnthropicBackend(cfg, client=fake).generate(Path(out))
        self.assertIn("Real-data reference", fake.last_user)

    def test_unsafe_config_fails_at_construction(self):
        cfg = {"backend": "anthropic_llm", "model": "m",
               "few_shot": {"real_data": str(self.d), "mode": "rows"}}
        with self.assertRaises(ConfigError):
            AnthropicBackend(cfg, client=Fake(["x"]))


if __name__ == "__main__":
    unittest.main()
