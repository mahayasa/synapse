# Roadmap and known gaps

## Not yet tested

- Real LLM calls with any provider (all LLM behaviour is tested with fake clients).
- Real public datasets (all evaluation so far used generator-made data).
- Platforms other than Windows with Python 3.13; GPU training and timing on larger data.

## Planned

- [ ] **Real data wins over the domain profile.** Add an explicit precedence rule to the LLM prompts and SKILL.md, list differences between a real sample and the profile in the hand-off, and warn in `run.py` when both `domain_profile` and `few_shot` are set.
- [ ] **Model adapter interface** so new GAN or diffusion models plug into the shared training pipeline ([extending.md](extending.md)).
- [ ] **Hyperparameter tuning** with the evaluator as the objective (`tune.py`).
- [ ] **Multi-config runner** that runs several configs and writes one leaderboard, closing the benchmark loop ([benchmark.md](benchmark.md)).
- [ ] **Omics evaluation metrics:** mean-variance trend, dispersion, library-size distribution, PCA structure, differential-expression recovery against the truth table.
- [ ] **Time-series evaluation metrics:** per-user autocorrelation, weekly volume patterns, sequence distance (for example DTW) on a time-based split.
- [ ] **Sequential models:** SDV's sequential synthesizer (PAR) is not wired in for sensor time series.
- [ ] **Conditional sampling** (for example generate only a given diagnosis) for CTGAN.
- [ ] **Differential-privacy training** option and a membership-inference test.
- [ ] **LLM fine-tuning** (open-weight models on serialised rows) as a research benchmark arm.
- [ ] **Expert review** of the clinical, physiological and biological values in the domain profiles, with citations.
- [ ] **Packaging** (`pyproject.toml`) and a continuous-integration matrix across Python versions and platforms.

## Contributing to these

See [CONTRIBUTING.md](../CONTRIBUTING.md).
