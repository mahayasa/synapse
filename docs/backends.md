# Backends and configuration

`run.py --config <file>` reads a YAML or JSON config, selects a backend, generates tables, validates them, optionally evaluates them against real data, and writes a run log. Example configs are in `configs/`.

## Output

Default folder: `out/<backend>_<model-or-algorithm>_seed<N>/` (override with `--out` or `out_dir`).

| File | When |
|---|---|
| `*.csv` | Always: the generated tables |
| `run_log.json` | Always: backend, seed, rows, timing, tokens (LLM), validation result, evaluation summary |
| `validation_report.txt` | Always |
| `generator.py` | LLM write-generator mode: the accepted script |
| `evaluation_report.md` / `.json` | When an `evaluation:` block with `real_data` is set |

The run log never contains API keys.

## Common keys

| Key | Meaning |
|---|---|
| `backend` | `rules`, `anthropic_llm`, `openai_llm`, `openai_compat`, `sdv` |
| `seed` | Random seed (default 42) |
| `rows` | Size of the main fact table |
| `out_dir` | Output folder |
| `domain`, `domain_profile`, `description`, `start_date` | Task text for LLM backends; `domain_profile` loads `references/<name>.md` |
| `validation` | `pk`, `fk`, `range`, `notnull` rules for `scripts/validate.py` (below) |
| `evaluation` | Score against real data (see [evaluation.md](evaluation.md)) |

### Validation rules

```yaml
validation:
  pk: [patients=patient_id, encounters=encounter_id]        # table=column[,column]
  fk: [encounters.patient_id=patients.patient_id]           # child.col=parent.col
  range: ["labs.value=0:", "vitals.sbp=40:250"]             # table.col=min:max (either side may be empty)
  notnull: [patients.patient_id]
```

## `rules`

Runs a seeded generator script. No model, no key.

| Key | Meaning |
|---|---|
| `generator` | Script path (default `scripts/generator_template.py`); must accept `--seed`, a rows argument, `--out` |
| `rows_arg` | Name of the rows argument (default `--orders`) |

## `anthropic_llm`, `openai_llm`, `openai_compat`

| Key | Meaning |
|---|---|
| `model` | Model id, for example `claude-sonnet-5-5`, or whatever your server calls the model |
| `mode` | `write_generator` (default) or `direct_rows` |
| `api_key_env` | **Name** of the environment variable that holds the key |
| `no_api_key` | `true` for local servers that need no key |
| `base_url` | Endpoint for OpenAI-compatible servers, for example `http://localhost:11434/v1` |
| `max_tokens`, `temperature` | Passed to the API |
| `max_tokens_param` | `max_tokens` (default) or `max_completion_tokens` for models that need it |
| `max_repairs` | Repair attempts after a failed script or validation (default 3) |
| `max_calls` | Hard limit on API calls (default 12) |
| `max_total_tokens` | Hard limit on tokens (default 400000) |
| `script_timeout` | Seconds a generated script may run (default 300) |

### `write_generator` mode

The model writes a seeded Python script (numpy and pandas only) that accepts `--seed`, `--rows`, `--out`. Synapse runs it, runs the validation rules, and on failure sends the error back for repair. Reproducible: the saved `generator.py` regenerates the data without the model.

### `direct_rows` mode

The model emits CSV rows in chunks (`rows_per_call`). It needs a `tables:` list:

```yaml
tables:
  - {name: patients, rows: 60, columns: "patient_id (P0001..), age, sex, bmi, outcome (0/1)"}
```

Slow, costly and **not reproducible**; suited to small tables and to the benchmark's direct-generation arm.

### API keys

- The config holds the variable name, for example `api_key_env: ANTHROPIC_API_KEY`.
- A value that looks like a pasted key is rejected without being printed.
- Keys are never written to logs, output files or the generated code's environment.
- Local servers can run with `no_api_key: true`.

### Few-shot context from real data

```yaml
few_shot:
  real_data: data/real
  mode: stats                   # stats (default) | rows
  rows_per_table: 5             # rows mode only
  tables: [patients]            # optional filter
  exclude_columns: [name, mrn, address]
  max_chars: 6000
  # acknowledge_external_data: true   # required for rows mode with a hosted model
```

- `real_data` is a folder or a single file; CSV, TSV, pasted-text and XLSX are read (each sheet is a table). See [example-data.md](example-data.md).
- `stats` sends per-column aggregates: types, quantiles, top categories (small integer codes and flags are shown as category mixes), masked formats for identifier-like columns.
- `rows` sends real example rows. With a hosted model it refuses to run unless `acknowledge_external_data: true`. A `localhost` `base_url` needs no flag.
- Everything built here is sent to the model provider. See [security-and-privacy.md](security-and-privacy.md).

### Real data versus the domain profile

If both `domain_profile` and `few_shot` are set, the prompt tells the model that the real-data reference wins where they disagree and asks it to list the differences in a comment at the top of the script; `run.py` also prints a warning and records it in `run_log.json` (`warnings`). This is an instruction to the model, not something code enforces. When your real data differs a lot from the profile, drop `domain_profile` or use the `sdv` backend. See [domains.md](domains.md).

## `sdv`

Gaussian copula, CTGAN and TVAE trained on real CSVs. See [training.md](training.md).

## Cost guards

`max_calls`, `max_total_tokens` and `max_repairs` stop runaway configurations. Exceeding a limit raises an error and stops the run. Token use is recorded in `run_log.json`.
