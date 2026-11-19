# Synapse

**Synthetic data for health and biomedical research.** Synapse is a config-driven agent skill and Python toolkit that generates, trains, validates and evaluates synthetic datasets, and lets you swap the generator between rule-based code, large language models and classical machine-learning models (CTGAN, TVAE, Gaussian copula) so they can be benchmarked against each other.

> **Status: research prototype (v0.1.0).** Everything here is tested offline with fake LLM clients and generator-made data. Real LLM calls and real clinical, genomic or wearable datasets have not been tested yet. Synthetic data produced by Synapse is **not** real patient data and must not be used for medical or biological conclusions.

## Why

Comparing synthetic-data generators fairly is hard: they need the same schema, the same validation and the same quality metrics. Synapse provides one harness where the model and the API-key variable are configuration, quality is measured against real data with standard methods, and the domain knowledge (EHR, RNA-seq, wearables) is kept in editable reference files.

## Features

- **Swappable backends:** rule-based generator, Claude (Anthropic), OpenAI and any OpenAI-compatible server (Ollama, vLLM, OpenRouter), and SDV models (Gaussian copula, CTGAN, TVAE). See [docs/backends.md](docs/backends.md).
- **LLM agent loop:** the model writes a seeded generator script, Synapse runs and validates it, and errors are fed back for repair. Cost guards limit calls, tokens and repairs.
- **Few-shot from real data:** show an LLM what real data looks like (aggregates or example rows) with explicit privacy guards.
- **Train, then sample:** fit CTGAN/TVAE/Gaussian copula once on real data, save the model, and sample any size later without the real data. Keys, foreign keys and derived columns are rebuilt so integrity holds by construction. See [docs/training.md](docs/training.md).
- **Evaluator:** fidelity (column shapes, pair trends, C2ST), utility (TSTR), privacy (distance to closest record and exact-copy rate), coverage and validity, and integrity, plus a leaderboard to compare runs. See [docs/evaluation.md](docs/evaluation.md).
- **Health and biomedical domain profiles:** EHR and disease classification, gene-level RNA-seq, and exercise and wearable data. See [docs/domains.md](docs/domains.md).
- **Works as a Claude skill:** drop the folder into `~/.claude/skills/synapse/` and use `/synapse`.

## Quick start

Requirements: Python 3.11+ (developed on Python 3.13 on Windows; CI also runs Python 3.11 and 3.13 on Linux).

```bash
git clone https://github.com/mahayasa/synapse.git
cd synapse
pip install -r requirements.txt            # numpy, pandas, scipy, scikit-learn, pyyaml
# optional, only for the backends you use:
pip install -r requirements-optional.txt   # anthropic, openai, sdv (sdv pulls in PyTorch)
```

Generate data with the rule-based backend (no model, no key) and validate it:

```bash
python run.py --config configs/rules.yaml --out out/demo
```

Run the offline test suite:

```bash
python -m unittest discover tests
```

Score synthetic data against real data, or compare several runs:

```bash
python evaluate.py --real data/real --synthetic out/run_a out/run_b --labels claude ctgan
```

Train a classical model on real data, then sample from it:

```bash
python train.py --config configs/ctgan.yaml
python run.py   --config configs/ctgan.yaml
```

Use an LLM backend (the config names the environment variable that holds your key, never the key itself):

```bash
# PowerShell:  $env:ANTHROPIC_API_KEY = "..."
# bash:        export ANTHROPIC_API_KEY="..."
python run.py --config configs/anthropic.yaml
```

A complete worked example (a small diabetes classification table) is in [examples/diabetes_classification/](examples/diabetes_classification/).

## Use as a Claude skill

```bash
git clone https://github.com/mahayasa/synapse.git ~/.claude/skills/synapse
```

Start a new Claude Code session and run, for example, `/synapse generate a small diabetes classification table`. Skills load at session start. See [docs/getting-started.md](docs/getting-started.md).

## How it fits together

```mermaid
flowchart LR
    C[config.yaml] --> R[run.py]
    R --> B{backend}
    B -->|rules| G1[seeded generator script]
    B -->|LLM| G2[model writes generator + repair loop]
    B -->|sdv| G3[saved CTGAN / TVAE / copula model]
    G1 --> O[CSV tables]
    G2 --> O
    G3 --> O
    O --> V[validate.py: PK, FK, ranges]
    O --> E[evaluate.py vs real data]
    P[references/*.md domain profiles] -.-> G2
    F[few-shot summary of real data] -.-> G2
    T[train.py on real data] --> G3
```

More detail: [docs/architecture.md](docs/architecture.md).

## Repository layout

```
synapse/
├── SKILL.md              # the agent skill definition (process and rules)
├── run.py                # config-driven runner
├── train.py              # train a classical model on real data
├── evaluate.py           # score synthetic vs real, compare runs
├── backends/             # rules, LLM (Anthropic/OpenAI/compatible), SDV, few-shot
├── evaluation/           # metric implementations and reports
├── scripts/              # generator template and validate.py
├── references/           # health_ehr, biomedical_omics, exercise_wearables
├── configs/              # example configs for each backend
├── tests/                # offline unit tests
├── examples/             # worked example
└── docs/                 # full documentation
```

## Documentation

| Document | Contents |
|---|---|
| [Getting started](docs/getting-started.md) | Install, first runs, use as a Claude skill, troubleshooting |
| [Backends and configuration](docs/backends.md) | Every config key, API keys, few-shot, cost guards |
| [Training classical models](docs/training.md) | Train/sample workflow, key handling, derived columns |
| [Evaluation](docs/evaluation.md) | Metric definitions, how to read scores, caveats |
| [Domain profiles](docs/domains.md) | The three profiles, real data vs profile, adding a domain |
| [Security and privacy](docs/security-and-privacy.md) | Keys, sandboxing, pickles, sensitive data |
| [Architecture](docs/architecture.md) | Data flow and module responsibilities |
| [Extending](docs/extending.md) | Adding a backend (GAN, diffusion) |
| [Benchmark plan](docs/benchmark.md) | Research design for comparing generators |
| [Roadmap](docs/roadmap.md) | Known gaps and next steps |

## Limitations

- Offline tests pass in CI on Linux (Python 3.11, 3.13) and Windows (3.13) without sdv. SDV training was tested only on Windows with Python 3.13, using generator-made data. No real LLM calls, no real public datasets, no GPU timing yet.
- The evaluator targets flat and relational tables. Metrics for gene-by-sample matrices and sensor time series are not implemented.
- Classical models are not byte-reproducible; saved models are pickles.
- The privacy score is a distance heuristic, not a formal guarantee.
- Clinical and physiological values in the domain profiles are rules of thumb and have not been checked against sources.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE).
