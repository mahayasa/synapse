# Changelog

## 0.2.0 (2026-11-19) - few-shot from example data

### Added
- `profile_data.py`: profile example data from CSV, TSV, pasted text (markdown, tab, comma, semicolon, pipe, aligned) or XLSX (one table per sheet); prints column summaries, likely keys and foreign keys, derived columns, strong correlations, and warnings about sample size, mostly-empty columns and possibly personal columns; `--export` and `--holdout` write normalised CSVs with a held-out part for evaluation.
- `backends/dataio.py` reader; the `few_shot` config now accepts a folder or a single file, including XLSX.
- `SKILL.md` workflow for example data given in a session: privacy confirmation, few-shot not copying, the sample wins over the domain profile, honest size limits, holdout evaluation.
- LLM prompts state that the real-data reference wins over the domain profile; `run.py` warns when both are set and records it in `run_log.json`.
- Flags and small integer codes are summarised as category mixes (class balance is visible).
- `docs/example-data.md`; `openpyxl` as an optional dependency; 12 new offline tests (38 in total).

### Known limitations
- Tested only with constructed samples, not real datasets; legacy `.xls` is not supported.

## 0.1.0 (2026-11-19) - initial release

First public version.

### Added
- Agent skill definition (`SKILL.md`) with a seven-step process and rules for reproducible, validated synthetic data.
- Config-driven runner (`run.py`) with swappable backends: `rules`, `anthropic_llm`, `openai_llm`, `openai_compat`, `sdv`.
- LLM write-generator mode with a repair loop, plus direct-rows mode; cost guards (`max_calls`, `max_total_tokens`, `max_repairs`).
- API keys read from a named environment variable; pasted keys are rejected; generated code runs with secrets stripped from its environment.
- Few-shot context from real data (`stats` or `rows` mode) with an explicit acknowledgement required for sending real rows to a hosted model.
- Train/sample workflow for Gaussian copula, CTGAN and TVAE (`train.py`): rebuilt primary and foreign keys, derived columns, held-out evaluation, loss logging.
- Evaluator (`evaluate.py`): column shapes, pair trends, C2ST, TSTR utility, DCR privacy with exact-copy rate, coverage and validity, integrity, composite score, and a comparison leaderboard.
- Domain profiles: `health_ehr`, `biomedical_omics`, `exercise_wearables`.
- Generic validator (`scripts/validate.py`) and a seeded generator template.
- 26 offline unit tests; worked diabetes classification example.

### Known limitations
- Offline tests pass in CI on Linux (Python 3.11, 3.13) and Windows (3.13); SDV training was tested only on Windows with Python 3.13; no real LLM calls or real datasets yet.
- No evaluator metrics yet for gene-by-sample matrices or time series.
- GAN training is not byte-reproducible; saved models are pickles.
