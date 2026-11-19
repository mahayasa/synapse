# Extending Synapse

## Add a backend today

A backend is a class with a `generate(out_dir)` method that writes CSV files and returns a metadata dictionary for the run log. Validation, evaluation, comparison and logging then work automatically.

1. Create `backends/my_backend.py`:

   ```python
   from pathlib import Path
   from .base import Backend, ConfigError

   class MyBackend(Backend):
       name = "my_backend"

       def label(self):                       # used in the output folder name
           return f"my_backend_{self.cfg.get('model', 'default')}"

       def generate(self, out_dir: Path) -> dict:
           out_dir.mkdir(parents=True, exist_ok=True)
           # ... produce DataFrames and write them to out_dir / "<table>.csv"
           return {"model": "...", "reproducible": False}   # never include secrets
   ```

2. Register it in `backends/__init__.py`:

   ```python
   REGISTRY = {..., "my_backend": ("backends.my_backend", "MyBackend")}
   ```

3. Add an example in `configs/` and a test in `tests/` (use fakes; keep tests offline).
4. Import heavy or optional packages lazily inside methods, raising `ConfigError("pip install ...")` if missing.
5. Document the backend in [backends.md](backends.md).

## A new GAN or diffusion model

Today the train/sample pipeline (key rebuilding, foreign keys, derived columns, held-out evaluation, `train.py`) lives in `backends/sdv_backend.py` and is tied to SDV synthesizers. For a one-off experiment you can write a standalone backend as above and handle keys yourself.

**Planned refactor:** a small adapter interface so any tabular model gets the whole pipeline for free:

```python
class ModelAdapter:
    def fit(self, df, params): ...      # learn from one table's non-key columns
    def sample(self, n): ...            # return a DataFrame
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

with a config such as `conventional: {adapter: "package.module:MyDiffusionAdapter", params: {...}}`. The adapter could live in your own research package so the skill does not need your model's dependencies.

Things to plan for with GAN and diffusion models:

- **Tabular scope.** The pipeline assumes one table at a time with mixed numeric and categorical columns. Matrices (gene by sample) and sequences (sensor series) need their own output format and metrics.
- **Compute.** Diffusion models usually need a GPU and train for much longer; record the device and time (the training log does this).
- **Reproducibility and persistence.** State whether the model is seeded and how it is saved. Pickles are unsafe to load from others.
- **Fair comparison.** Record hyperparameters and tuning effort for every model in a benchmark.

## Add an evaluation metric

Add a function in `evaluation/metrics.py` returning a score in 0..1 (plus details), call it from `evaluation/report.py`, add it to a group and, if it is a new group, to `WEIGHTS`. Add a test with a known-good and a known-bad case (see `tests/test_evaluation.py`) and document it in [evaluation.md](evaluation.md).

## Add a domain profile

See [domains.md](domains.md).
