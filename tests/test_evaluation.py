"""Evaluator tests on known-good / known-bad / copied synthetic data (offline)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluation import compare, evaluate_dataset, to_markdown  # noqa: E402
from generator_template import CONFIG, gen_customers, gen_orders, gen_products, make_rng  # noqa: E402

CFG = {"seed": 0, "utility": {"table": "orders", "target": "total", "task": "regression"}}
VALID = {"pk": ["orders=order_id"]}


def build(root: Path):
    rng = make_rng(1)
    c, p = gen_customers(rng, CONFIG), gen_products(rng, CONFIG)
    o = gen_orders(rng, CONFIG, c, p, 8000).sample(frac=1, random_state=0).reset_index(drop=True)
    real, good = o.iloc[:4000].copy(), o.iloc[4000:].copy()
    bad = good.copy()
    r2 = np.random.default_rng(5)
    bad["qty"] = r2.integers(1, 21, len(bad))
    bad["unit_price"] = r2.uniform(1, 5000, len(bad)).round(2)
    bad["total"] = r2.normal(50000, 9000, len(bad)).round(2)
    bad["is_anomaly"] = r2.random(len(bad)) < 0.5
    copy = real.copy()
    for name, df in {"real": real, "good": good, "bad": bad, "copy": copy}.items():
        (root / name).mkdir()
        df.to_csv(root / name / "orders.csv", index=False)


class EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        build(root)
        cls.rep = {k: evaluate_dataset(root / "real", root / k, CFG, VALID) for k in ("good", "bad", "copy")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tbl(self, k):
        return self.rep[k]["tables"]["orders"]

    def test_good_beats_bad_on_every_fidelity_metric(self):
        g, b = self.tbl("good"), self.tbl("bad")
        self.assertGreater(g["column_shapes"]["score"], 0.9)
        self.assertGreater(g["column_shapes"]["score"], b["column_shapes"]["score"] + 0.2)
        self.assertGreater(g["pair_trends"]["score"], b["pair_trends"]["score"])
        self.assertGreater(g["c2st"]["score"], 0.8)
        self.assertLess(b["c2st"]["score"], 0.2)          # detector spots the fake easily
        self.assertGreater(self.rep["good"]["composite"], self.rep["bad"]["composite"])

    def test_copy_is_flagged_by_privacy(self):
        c, g = self.tbl("copy")["privacy"], self.tbl("good")["privacy"]
        self.assertGreater(c["exact_match_rate_synthetic"], 0.7)
        self.assertLess(c["score"], 0.2)
        self.assertGreater(g["score"], 0.8)

    def test_utility_tstr(self):
        self.assertGreater(self.tbl("good")["utility"]["score"], 0.8)
        self.assertTrue(self.tbl("bad")["utility"]["score"] is None or self.tbl("bad")["utility"]["score"] < 0.5)

    def test_ids_excluded_and_integrity_scored(self):
        self.assertNotIn("customer_id", self.tbl("good")["columns_evaluated"])
        self.assertNotIn("order_id", self.tbl("good")["columns_evaluated"])
        self.assertEqual(self.rep["good"]["groups"]["integrity"], 1.0)

    def test_reports_render(self):
        md = to_markdown(self.rep["good"])
        self.assertIn("column_shapes", md)
        self.assertIn("good", compare(self.rep))

    def test_deterministic(self):
        root = Path(self.tmp.name)
        again = evaluate_dataset(root / "real", root / "good", CFG, VALID)
        self.assertEqual(again["composite"], self.rep["good"]["composite"])


if __name__ == "__main__":
    unittest.main()
