---
name: synapse
description: Generate realistic, reproducible synthetic datasets (relational tables, time series, transactions) for demos, testing, dashboards, and ML prototyping. Use when the user asks for fake/mock/dummy/sample/synthetic data, a seed dataset, or test fixtures, especially for health and biomedical topics: EHR or disease-classification data (e.g. diabetes), gene-level data (RNA-seq), or exercise/wearable data (smartwatch, sport). Covers schema design, seasonality, skew, correlations, injected anomalies, model training, and evaluation.
---

# Synapse: synthetic data for health and biomedical research

Produce data that *behaves* like the real domain: right shapes, right messiness, fully reproducible, and verified before hand-off.

## Process

1. **Clarify the purpose and scope.** Ask only what's missing: what the data is for (demo, load test, ML, dashboard), row volume, date range, output format (CSV / Parquet / SQL inserts), and whether it must match an existing schema. Defaults: CSV, seed 42, 12 months, ~10k fact rows.
2. **Load a domain profile** from `references/`: `health_ehr.md` (EHR, disease classification such as diabetes), `biomedical_omics.md` (gene-level data such as RNA-seq), or `exercise_wearables.md` (smartwatch/sport data). Each also says which backend suits the domain and which evaluation metrics apply. If none fits, build a profile in step 3 from the user's description and offer to save it as a new one.
3. **Design the schema.** List entities, primary keys, foreign keys, and the grain of each fact table. Write it down (a short table is enough) and confirm with the user if the design is non-obvious. Dimensions first, facts second.
4. **Choose the behaviours** from the domain profile's "Realism features" section: seasonality, trend, skewed distributions, correlated columns, missing data, and deliberately injected anomalies. Every realism choice should be a named parameter, not a magic number buried in code.
5. **Write the generator** by copying `scripts/generator_template.py`. Keep it seeded (one `numpy.random.Generator` passed everywhere), generate dimensions before facts, and derive dependent columns from their parents (e.g. `line_total = qty * unit_price`) instead of sampling them independently.
6. **Validate.** Run `scripts/validate.py` with the PK, FK, range, and not-null rules for this dataset. Fix the generator, not the output, until it passes. Then check the domain-specific sanity rules from the profile (e.g. expiry after manufacture date).
6b. **Evaluate (when real data exists).** Quality can only be measured against real data. If the user has a real dataset (or a reference sample), run `python evaluate.py --real <dir> --synthetic <out_dir>` or add an `evaluation:` block to the run config. Report the group scores and the weakest columns, not just the composite. Without real data, say so: only integrity and self-consistency can be checked.
7. **Hand off.** Report: files and row counts, the seed, the realism features included, injected anomalies (and how to find them via the `is_anomaly` flag), and how to regenerate. Never imply the data is real or de-identified real data.

## Rules

- **Reproducible:** same seed and config must give byte-identical output. No `random`/`numpy` global state, no `datetime.now()` in generated values (use a fixed `--end-date`).
- **No real people:** invent names/IDs from small seeded pools; never scrape or paste real personal data. Use obviously fake formats for emails (`@example.com`) and phone numbers.
- **Integrity first:** zero orphan foreign keys, unique primary keys, no unintended nulls. Intentional dirt (nulls, outliers, duplicates) must be configurable and flagged.
- **Label anomalies:** injected anomalies carry a boolean `is_anomaly` (or a separate `anomalies.csv`) so downstream detection can be scored.
- **Scale check:** generate a small sample first (`--rows 500`), inspect it, then scale up.
- **Medical/financial domains:** data is synthetic only. Don't encode real dosing guidance or real patient-level patterns as fact.

## Choosing a model / backend

`run.py` generates data through a swappable backend picked in a config file (`configs/*.yaml`):

| `backend` | What it does | Needs |
|---|---|---|
| `rules` | Runs a seeded generator script | nothing |
| `anthropic_llm` | Claude writes a generator (`mode: write_generator`) or emits rows (`direct_rows`) | `anthropic` pkg, key env var |
| `openai_llm` / `openai_compat` | OpenAI or any OpenAI-compatible server (Ollama, vLLM, OpenRouter, Gemini's compat endpoint) | `openai` pkg, key env var or `no_api_key: true` |
| `sdv` | Gaussian copula / CTGAN / TVAE: train on real CSVs (`train.py`), sample from the saved model | `sdv` pkg, real data to train |

Run: `python run.py --config configs/anthropic.yaml`. Output goes to `out/<backend>_<model>_seed<N>/` with `run_log.json` and `validation_report.txt`.

**API keys:** put the *name* of the environment variable in `api_key_env`, never the key. Ask the user to set the variable in their own shell; never request, print, or store a key. Cost guards: `max_calls`, `max_total_tokens`, `max_repairs`.

When the user names a model or says "use Claude/GPT/CTGAN", copy the closest `configs/*.yaml`, change `backend`/`model`/`domain`, and run it. If they ask for a conventional model but have no real data, say it can't run and offer `rules` or an LLM backend instead. Generated LLM code runs with secrets stripped from its environment but is not truly sandboxed.

## Few-shot for LLM backends

Add a `few_shot:` block (see `configs/anthropic_fewshot.yaml`) so the model sees what real data looks like. `mode: stats` (default) sends per-column aggregates and masked ID formats; `mode: rows` sends real example rows. **Whatever it builds is sent to the model provider.** `rows` mode with a hosted model requires `acknowledge_external_data: true` (not needed for a localhost `base_url`); only set that flag if the user has said sending real rows is fine. Use `exclude_columns` for sensitive fields, and prefer `stats` or a local model for sensitive data.

## Training classical models (CTGAN / TVAE / Gaussian copula)

Needs REAL data and `pip install sdv`. Two steps, configured in `configs/ctgan.yaml`:

1. `python train.py --config configs/ctgan.yaml`: fits per-table models, saves them to `model_dir` with `schema.json`, `training_log.json` (hyperparameters, device, loss curves) and `holdout_evaluation.md` (scored on held-out real rows; holdout copies are deleted).
2. `python run.py --config configs/ctgan.yaml`: samples any size from the saved model, no real data needed. `run.py` trains first if no model exists.

Design: primary/foreign keys are not modelled; PKs are rebuilt in the real ID format and FKs are drawn from parent keys with the real child-count skew, so integrity holds by construction. `derived` expressions (e.g. `orders.total: "qty * unit_price"`) are recomputed after sampling. `rows` sets the largest fact table; dimension tables keep their real size unless listed in `scale_tables`. Tune with `params` (epochs, batch_size, ...) and read the held-out evaluation: short training on small data gives low fidelity. Not byte-reproducible. Saved models are pickles: load only your own. On Windows the runner sets `JOBLIB_MULTIPROCESSING=0` to avoid a CTGAN multiprocessing crash.

## Evaluation

`evaluate.py` / `evaluation/` implement standard synthetic-data evaluation. All scores are 0-1, higher is better:

| Group | Method |
|---|---|
| fidelity | column shapes (1-KS / 1-TVD), pair trends (Spearman gap / joint TVD), C2ST (classifier real-vs-synthetic AUC) |
| utility | TSTR: train on synthetic, test on real, relative to train-on-real (needs `--utility table:target[:task]`) |
| privacy | DCR vs a real holdout baseline plus exact-copy rate, so copying real rows scores low |
| coverage_validity | boundary adherence, range and category coverage, novel categories |
| integrity | share of PK/FK/range checks passed (from the run's `validation` section) |

The composite is a heuristic weighted mean (weights in `evaluation/report.py`); always cite the groups. Compare runs: `python evaluate.py --real data/real --synthetic out/a out/b --labels claude ctgan` prints a leaderboard and writes `evaluation_report.md/json` into each folder.

Caveats to state when reporting: scores are noisy with under ~500 real rows; identifier columns (`id`, `*_id`, near-unique columns) are ignored; the privacy score is a distance heuristic, not a formal guarantee; a high composite does not prove the data is safe or realistic for a use the metrics don't cover.

## Files

- `train.py`: train and save a classical model. `evaluate.py` / `evaluation/`: score synthetic vs real.
- `run.py`, `backends/`, `configs/`, `tests/`: the config-driven runner, its backends, example configs, and offline tests (`python -m unittest discover tests`).
- `scripts/generator_template.py`: seeded skeleton with dimensions, seasonal fact generation, skewed assignment, anomaly injection.
- `scripts/validate.py`: generic checks (PK uniqueness, FK integrity, nulls, duplicates, ranges). Run `python validate.py --help`.
- `references/health_ehr.md`: EHR and disease-classification profile (patients, encounters, labs; diabetes-style tables).
- `references/biomedical_omics.md`: gene-level RNA-seq profile (NB counts, DE truth, batch effects).
- `references/exercise_wearables.md`: smartwatch and sport profile (HR dynamics, workouts, daily and sleep summaries).
- Add more `<domain>.md` profiles over time using the same headings.

## Adding a domain profile

Copy the section headings of `references/health_ehr.md`: Entities, Realism features, Sanity rules, Typical volumes, Choosing a backend and evaluation notes, Pitfalls.

Health and biomedical data are sensitive: never send real patient, genomic or location data to a hosted model, and describe results as synthetic, not clinically or biologically valid.
