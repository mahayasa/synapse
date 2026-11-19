# Getting started

## Requirements

- Python 3.11 or newer (developed and tested on Python 3.13 on Windows).
- Core packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyyaml` (`pip install -r requirements.txt`).
- Optional, per backend: `anthropic`, `openai`, `sdv` (`pip install -r requirements-optional.txt`). `sdv` installs PyTorch and is a large download.

## Install

```bash
git clone https://github.com/mahayasa/synapse.git
cd synapse
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1      bash: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover tests
```

All tests should pass. Tests that need `sdv` are skipped if it is not installed.

## First run: rule-based backend

The `rules` backend needs no model and no key. It runs the seeded template in `scripts/generator_template.py`:

```bash
python run.py --config configs/rules.yaml --out out/demo
```

Output in `out/demo/`:

| File | Contents |
|---|---|
| `customers.csv`, `products.csv`, `orders.csv` | The generated tables |
| `run_log.json` | Backend, seed, rows, timing, validation result |
| `validation_report.txt` | PK, FK, range and not-null checks |

Run the command twice with the same seed: the CSVs are byte-identical.

## Run with an LLM

1. Install the package: `pip install anthropic` (or `openai`).
2. Set the key in your own shell. The config only names the variable:
   ```bash
   # PowerShell:  $env:ANTHROPIC_API_KEY = "..."
   # bash:        export ANTHROPIC_API_KEY="..."
   ```
3. Run: `python run.py --config configs/anthropic.yaml`

The model writes a generator script, Synapse runs and validates it, and any error is sent back for repair (up to `max_repairs`). The accepted script is saved as `generator.py` in the output folder so the run can be reproduced without the model.

For a local model, use `configs/openai_compat.yaml` with an Ollama or vLLM server and no key. See [backends.md](backends.md).

## Train a classical model on real data

```bash
pip install sdv
python train.py --config configs/ctgan.yaml    # fit once, save to model_dir
python run.py   --config configs/ctgan.yaml    # sample from the saved model
```

Put your real CSVs in `data/real/` (git-ignored). See [training.md](training.md).

## Evaluate against real data

```bash
python evaluate.py --real data/real --synthetic out/run_a out/run_b --labels claude ctgan
```

See [evaluation.md](evaluation.md).

## Use as a Claude skill

Place the repository at `~/.claude/skills/synapse/` (or `.claude/skills/synapse/` inside a project):

```bash
git clone https://github.com/mahayasa/synapse.git ~/.claude/skills/synapse
```

Claude loads skills when a session starts, so start a new session. Then either type `/synapse` followed by a request, or ask in plain language, for example "generate a small diabetes classification table". Claude follows the process in `SKILL.md`: it loads the matching domain profile, designs the schema, generates, validates, and reports.

In Claude.ai, zip the folder with `SKILL.md` at the top level and upload it under Settings, Capabilities, Skills (requires a plan with Skills and code execution).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: pip install anthropic` | Optional package missing; install it |
| `Environment variable ... is not set` | Set the variable named in `api_key_env` in your shell |
| `api_key_env must be an environment variable NAME` | You pasted a key into the config. Remove it and rotate the key if it was ever committed |
| `sdv backend needs ... real_data` | Conventional models cannot learn from a description; provide real CSVs or use another backend |
| CTGAN crashes with `_posixsubprocess` on Windows | The runner sets `JOBLIB_MULTIPROCESSING=0` automatically; if you run the library directly, set it yourself |
| Skill not found in Claude | Start a new session; check the folder is `~/.claude/skills/synapse/SKILL.md` |
