"""Train/sample workflow tests. Skipped if sdv is not installed. Tiny data, few epochs, CPU."""
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HAVE_SDV = importlib.util.find_spec("sdv") is not None

if HAVE_SDV:
    from backends.sdv_backend import sample_tables, train  # noqa: E402
    from run import run  # noqa: E402


def make_real(d: Path):
    rng = np.random.default_rng(0)
    n_c, n_o = 150, 600
    c = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(1, n_c + 1)],
                      "segment": rng.choice(["retail", "member", "wholesale"], n_c, p=[.7, .25, .05]),
                      "age": rng.normal(40, 12, n_c).clip(18, 90).round()})
    w = 1 / np.arange(1, n_c + 1) ** 1.1
    cust = rng.choice(c["customer_id"], n_o, p=w / w.sum())          # skewed activity
    qty = rng.integers(1, 5, n_o)
    price = rng.lognormal(3, .4, n_o).round(2)
    o = pd.DataFrame({"order_id": [f"O{i:06d}" for i in range(1, n_o + 1)], "customer_id": cust,
                      "qty": qty, "unit_price": price, "total": (qty * price).round(2)})
    c.to_csv(d / "customers.csv", index=False)
    o.to_csv(d / "orders.csv", index=False)


def cfg_for(real: Path, model_dir: Path, algo="gaussian_copula", params=None):
    return {"backend": "sdv", "seed": 7, "rows": 800,
            "conventional": {"algorithm": algo, "real_data": str(real), "model_dir": str(model_dir),
                             "params": params or {}, "holdout_fraction": 0.2,
                             "primary_keys": {"customers": "customer_id", "orders": "order_id"},
                             "foreign_keys": ["orders.customer_id=customers.customer_id"],
                             "derived": {"orders.total": "qty * unit_price"}},
            "validation": {"pk": ["customers=customer_id", "orders=order_id"],
                           "fk": ["orders.customer_id=customers.customer_id"]}}


@unittest.skipUnless(HAVE_SDV, "sdv not installed")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.real = cls.root / "real"
        cls.real.mkdir()
        make_real(cls.real)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_train_saves_artifacts_and_scores_holdout(self):
        md = self.root / "m_gc"
        log = train(cfg_for(self.real, md), md)
        for f in ("schema.json", "training_log.json", "customers.pkl", "orders.pkl", "holdout_evaluation.md"):
            self.assertTrue((md / f).exists(), f)
        self.assertEqual(log["rows_held_out"], {"orders": 120})   # tables under 200 rows are not split
        self.assertIsNotNone(log["holdout_evaluation"]["composite"])
        self.assertFalse(any(p.name.startswith("holdout") and p.is_dir() for p in md.iterdir()))  # no real copies kept

    def test_sampling_without_real_data_keeps_integrity(self):
        md = self.root / "m_gc2"
        train(cfg_for(self.real, md), md)
        hidden = self.root / "real_hidden"
        shutil.move(str(self.real), str(hidden))                   # real data gone: sampling must still work
        try:
            t = sample_tables(md, rows=1000, seed=3)
        finally:
            shutil.move(str(hidden), str(self.real))
        c, o = t["customers"], t["orders"]
        self.assertEqual(len(o), 1000)
        self.assertTrue(c["customer_id"].is_unique and o["order_id"].is_unique)
        self.assertTrue(o["customer_id"].isin(c["customer_id"]).all())          # zero orphans
        self.assertTrue(c["customer_id"].str.fullmatch(r"C\d{5}").all())         # real ID format kept
        self.assertTrue(np.allclose(o["total"], (o["qty"] * o["unit_price"]).round(2)))  # derived recomputed
        self.assertGreater(o["customer_id"].value_counts().iloc[0], len(o) / len(c) * 2)  # skew preserved

    def test_run_py_end_to_end_trains_then_reuses_model(self):
        md = self.root / "m_run"
        out1, out2 = self.root / "o1", self.root / "o2"
        l1 = run(cfg_for(self.real, md), out1)
        self.assertTrue(l1["validation_passed"])
        self.assertTrue(l1["trained_this_run"])
        l2 = run(cfg_for(self.real, md), out2)
        self.assertFalse(l2["trained_this_run"])
        self.assertTrue(l2["validation_passed"])

    def test_ctgan_logs_loss_curve(self):
        md = self.root / "m_ct"
        log = train(cfg_for(self.real, md, "ctgan", {"epochs": 2, "batch_size": 100}), md)
        self.assertIn("orders", log["loss"])
        self.assertEqual(len(next(iter(log["loss"]["orders"].values()))), 2)
        self.assertIn(log["device"], ("cpu", "cuda"))


if __name__ == "__main__":
    unittest.main()
