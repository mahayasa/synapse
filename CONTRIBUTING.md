# Contributing

Thanks for your interest in Synapse. This is a research prototype; small, well-tested changes are the easiest to review.

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt   # only if you work on the LLM or SDV backends
python -m unittest discover tests
```

Tests are offline: LLM backends are tested with fake clients, so no API keys or network are needed. SDV tests are skipped automatically when `sdv` is not installed.

## Guidelines

- **Never commit secrets or real data.** API keys belong in environment variables; real datasets belong in `data/real/` (git-ignored). Do not add patient, genomic or location data of real people to the repository, including in tests and examples.
- **Keep generation reproducible.** Use a single seeded `numpy.random.Generator`; no global random state and no `datetime.now()` in generated values.
- **Add a test** for each behaviour change. Prefer small synthetic fixtures built inside the test.
- **Keep optional dependencies optional.** Import `anthropic`, `openai` and `sdv` lazily so the core runs without them.
- **Document user-facing changes** in `docs/` and `CHANGELOG.md`.
- **Be honest in docs:** state what was tested and what was not.

## Adding a backend or a domain profile

- A new generator: see [docs/extending.md](docs/extending.md).
- A new domain: copy the headings of `references/health_ehr.md` (Entities, Realism features, Sanity rules, Typical volumes, Choosing a backend and evaluation notes, Pitfalls) and cite sources for any clinical or physiological constants.

## Pull requests

Describe what changed and why, how you tested it, and anything you could not test (platform, Python version, real LLM calls, GPU).
