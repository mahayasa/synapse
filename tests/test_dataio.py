"""Example-data ingestion, profiling and the few-shot precedence rule (offline)."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import ConfigError  # noqa: E402
from backends.dataio import parse_text_table, read_tables, safe_name  # noqa: E402
from backends.fewshot import build_few_shot  # noqa: E402
from backends.llm import AnthropicBackend  # noqa: E402
from profile_data import build_profile, derived_relations, sensitive_columns  # noqa: E402
from test_backends import GOOD, Fake  # noqa: E402

HAVE_XLSX = importlib.util.find_spec("openpyxl") is not None


def make_tables(n=300):
    rng = np.random.default_rng(0)
    pat = pd.DataFrame({"patient_id": [f"P{i:04d}" for i in range(1, n + 1)],
                        "first_name": [f"Person{i}" for i in range(n)],
                        "age": rng.integers(20, 80, n), "bmi": rng.normal(27, 4, n).round(1),
                        "glucose": rng.normal(100, 15, n).round()})
    pat["hba1c"] = (4 + 0.02 * pat["glucose"] + rng.normal(0, 0.1, n)).round(1)
    m = 2 * n
    qty, price = rng.integers(1, 6, m), rng.uniform(5, 90, m).round(2)
    orders = pd.DataFrame({"order_id": range(1, m + 1), "patient_id": rng.choice(pat["patient_id"], m),
                           "qty": qty, "unit_price": price, "total": (qty * price).round(2)})
    return {"patients": pat, "orders": orders}


class ParseTests(unittest.TestCase):
    def test_delimiters_and_markdown(self):
        want = ["id", "age", "sex"]
        cases = {
            "csv": "id,age,sex\n1,30,F\n2,41,M\n",
            "tsv": "id\tage\tsex\n1\t30\tF\n2\t41\tM\n",
            "semicolon": "id;age;sex\n1;30;F\n2;41;M\n",
            "pipe": "id | age | sex\n1 | 30 | F\n2 | 41 | M\n",
            "markdown": "| id | age | sex |\n|:---|---:|---|\n| 1 | 30 | F |\n| 2 | 41 | M |\n",
            "aligned": "id  age  sex\n1   30   F\n2   41   M\n",
        }
        for name, text in cases.items():
            df = parse_text_table(text)
            self.assertEqual(list(df.columns), want, name)
            self.assertEqual(len(df), 2, name)
            self.assertEqual(int(df["age"].iloc[1]), 41, name)

    def test_bad_text(self):
        with self.assertRaises(ConfigError):
            parse_text_table("just one line")
        with self.assertRaises(ConfigError):
            parse_text_table("hello\nworld")

    def test_read_tables_file_and_folder_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "Patient Data.csv").write_text("a,b\n1,2\n3,4\n")
            (d / "pasted.txt").write_text("x\ty\n1\t2\n")
            (d / "notes.md").write_text("ignored in folder mode")
            self.assertEqual(list(read_tables(d / "Patient Data.csv")), ["patient_data"])
            self.assertEqual(list(read_tables(d / "pasted.txt")), ["pasted"])
            self.assertEqual(list(read_tables(d)), ["patient_data"])      # .txt/.md only when named
            (d / "old.xls").write_bytes(b"x")
            with self.assertRaises(ConfigError):
                read_tables(d / "old.xls")
            with self.assertRaises(ConfigError):
                read_tables(d / "missing.csv")
        self.assertEqual(safe_name("Visits Log (2025)"), "visits_log_2025")

    @unittest.skipUnless(HAVE_XLSX, "openpyxl not installed")
    def test_xlsx_sheets_become_tables(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "study.xlsx"
            with pd.ExcelWriter(p) as w:
                pd.DataFrame({"id": [1, 2], "v": [0.5, 0.7]}).to_excel(w, sheet_name="Patients", index=False)
                pd.DataFrame({"id": [1, 1], "x": [3, 4]}).to_excel(w, sheet_name="Visit Log", index=False)
                pd.DataFrame().to_excel(w, sheet_name="Empty", index=False)
            t = read_tables(p)
            self.assertEqual(sorted(t), ["patients", "visit_log"])      # empty sheet dropped
            self.assertEqual(len(t["patients"]), 2)

    @unittest.skipUnless(HAVE_XLSX, "openpyxl not installed")
    def test_fewshot_accepts_a_single_xlsx(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.xlsx"
            pd.DataFrame({"age": range(20, 70), "sex": ["F", "M"] * 25}).to_excel(p, index=False)
            text = build_few_shot({"few_shot": {"real_data": str(p)}})
            self.assertIn("Table `s`", text)
            self.assertIn("sex (category", text)


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = make_tables()
        cls.prof = build_profile(cls.tables)

    def test_keys_foreign_keys_derived_and_sensitive(self):
        self.assertIn("patient_id", self.prof["tables"]["patients"]["keys"])
        self.assertIn("order_id", self.prof["tables"]["orders"]["keys"])
        self.assertIn("orders.patient_id -> patients.patient_id", self.prof["foreign_keys"])
        derived = [e for e, _ in self.prof["tables"]["orders"]["derived"]]
        self.assertIn("total = qty * unit_price", derived)
        self.assertEqual([c for c, _ in sensitive_columns(self.tables["patients"])], ["first_name"])
        self.assertTrue(any("first_name" in w for w in self.prof["warnings"]))

    def test_correlation_detected_and_no_false_derived(self):
        pairs = [p for p, _ in self.prof["tables"]["patients"]["correlations"]]
        self.assertTrue(any("glucose" in p and "hba1c" in p for p in pairs))
        rng = np.random.default_rng(1)
        indep = pd.DataFrame({c: rng.normal(50, 10, 200) for c in "abcd"})
        self.assertEqual(derived_relations(indep), [])

    def test_size_notes(self):
        small = build_profile({"t": make_tables()["patients"].head(10)})
        self.assertIn("very small", small["tables"]["t"]["size_note"])
        self.assertIn("adequate", build_profile({"t": make_tables(600)["patients"]})["tables"]["t"]["size_note"])

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "profile_data.py"), *args],
                              capture_output=True, text=True)

    def test_cli_stats_mode_hides_rows_and_rows_mode_shows_them(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "patients.csv"
            self.tables["patients"].to_csv(p, index=False)
            r = self.run_cli(str(p))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("Table `patients`", r.stdout)
            self.assertIn("possible personal data", r.stdout)
            self.assertNotIn("Person17", r.stdout)                     # stats mode: no raw values
            r2 = self.run_cli(str(p), "--mode", "rows", "--rows", "3", "--exclude", "first_name")
            self.assertIn("Example rows", r2.stdout)
            self.assertNotIn("first_name", r2.stdout)                  # excluded column gone
            self.assertNotIn("possible personal data", r2.stdout)

    def test_cli_holdout_export(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "in").mkdir()
            self.tables["patients"].to_csv(d / "in" / "patients.csv", index=False)       # 300 rows: split
            self.tables["patients"].head(50).to_csv(d / "in" / "tiny.csv", index=False)  # 50 rows: not split
            r = self.run_cli(str(d / "in"), "--export", str(d / "out"), "--holdout", "0.3")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len(pd.read_csv(d / "out" / "holdout" / "patients.csv")), 90)
            self.assertEqual(len(pd.read_csv(d / "out" / "train" / "patients.csv")), 210)
            self.assertFalse((d / "out" / "holdout" / "tiny.csv").exists())
            self.assertEqual(len(pd.read_csv(d / "out" / "train" / "tiny.csv")), 50)
            self.assertIn("(profiled above)", r.stdout)
            self.assertIn("210", r.stdout)                              # profile is of the train part
            bad = self.run_cli(str(d / "in"), "--holdout", "0.3")
            self.assertNotEqual(bad.returncode, 0)

    def test_cli_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "orders.csv"
            self.tables["orders"].to_csv(p, index=False)
            r = self.run_cli(str(p), "--json")
            data = json.loads(r.stdout)
            self.assertEqual(data["tables"]["orders"]["rows"], 600)


class PrecedenceTests(unittest.TestCase):
    def test_profile_plus_fewshot_warns_and_prompt_states_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "real").mkdir()
            make_tables()["patients"].to_csv(d / "real" / "patients.csv", index=False)
            fake = Fake([GOOD])
            cfg = {"backend": "anthropic_llm", "model": "m", "rows": 20, "seed": 1,
                   "domain_profile": "health_ehr", "few_shot": {"real_data": str(d / "real")},
                   "validation": {"pk": ["orders=order_id"]}}
            meta = AnthropicBackend(cfg, client=fake).generate(d / "out")
            self.assertTrue(any("domain_profile" in w for w in meta["warnings"]))
            self.assertIn("Real-data reference", fake.last_user)
            self.assertIn("Domain profile", fake.last_user)
            cfg2 = {k: v for k, v in cfg.items() if k != "domain_profile"}
            meta2 = AnthropicBackend(cfg2, client=Fake([GOOD])).generate(d / "out2")
            self.assertEqual(meta2["warnings"], [])


if __name__ == "__main__":
    unittest.main()
