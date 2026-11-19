# Training classical models

The `sdv` backend trains Gaussian copula, CTGAN or TVAE models on **real** data, saves them, and samples from the saved models. Conventional models cannot learn from a description alone; without real CSVs they fail with a clear error.

```bash
pip install sdv
python train.py --config configs/ctgan.yaml    # fit once, save to model_dir
python run.py   --config configs/ctgan.yaml    # sample any size, no real data needed
```

`run.py` trains first if `model_dir` has no saved model.

## Configuration

```yaml
backend: sdv
conventional:
  algorithm: ctgan              # gaussian_copula | ctgan | tvae
  real_data: data/real          # folder of real CSVs (git-ignored)
  model_dir: models/ctgan
  params: {epochs: 300, batch_size: 500}   # passed to the synthesizer
  holdout_fraction: 0.2
  primary_keys: {patients: patient_id}
  foreign_keys: [encounters.patient_id=patients.patient_id]
  derived: {patients.bmi: "weight_kg / (height_cm / 100) ** 2"}
  scale_tables: []
seed: 42
rows: 10000
```

| Key | Meaning |
|---|---|
| `algorithm` | Which SDV synthesizer to use |
| `real_data` | Folder of real CSVs, one file per table (file name = table name) |
| `model_dir` | Where models are saved; `run.py` loads them from here |
| `params` | Hyperparameters passed to the synthesizer (epochs, batch size, ...) |
| `holdout_fraction` | Share of rows held out and scored after training. Tables under 200 rows are not split |
| `primary_keys` | `{table: column}`, rebuilt after sampling |
| `foreign_keys` | `child.col=parent.col`, rebuilt after sampling |
| `derived` | `{table.col: pandas-eval expression}` recomputed after sampling |
| `scale_tables` | Dimension tables that should scale with `rows` |

## How it works

1. **Key columns are not modelled.** Primary and foreign key columns are dropped before fitting, because a generative model cannot reproduce identifiers or join structure.
2. **Training.** Each table is fitted independently on its non-key columns. A fraction is held out first.
3. **Saved artifacts** in `model_dir`:
   - `<table>.pkl`: the fitted synthesizer
   - `schema.json`: columns, key formats, foreign-key child-count distributions, derived expressions (no real rows)
   - `training_log.json`: algorithm, hyperparameters, seed, device, training time, row counts, loss curves (CTGAN and TVAE), held-out evaluation summary
   - `holdout_evaluation.md`: the full held-out report. The held-out copies of real rows are deleted after scoring
4. **Sampling.** Tables are sampled parents first:
   - primary keys are rebuilt in the real ID format (for example `P00001`)
   - foreign keys are drawn from the parent's synthetic keys, weighted by the real per-parent child-count distribution so skew (a few very active parents) is preserved
   - `derived` expressions are recomputed, because GANs do not preserve arithmetic
   - so primary-key uniqueness and foreign-key integrity hold by construction
5. **Scaling.** `rows` sets the size of the largest fact table. Dimension tables (tables other tables point to) keep their real size unless listed in `scale_tables`.

## Reading the held-out report

A short run on little data gives low fidelity; in a test, CTGAN trained for 40 epochs on 2,000 rows scored about 0.53 on fidelity (a classifier separated real from synthetic with AUC 0.93). Increase `epochs` and data, then compare held-out reports. Scores on small holdouts are noisy.

## Limitations

- Not byte-reproducible: GAN training varies between runs even with a seed. Save the model and log its hyperparameters.
- Saved models are pickles: load only models you trained yourself.
- Tables are modelled independently; relationships between non-key columns of different tables are not learned.
- Suited to flat or relational tabular data. Not suited to gene-by-sample matrices (far more columns than rows) or sensor time series; use a parametric generator for those (see [domains.md](domains.md)).
- On Windows the runner sets `JOBLIB_MULTIPROCESSING=0` because CTGAN's process pool crashed on the tested Python build once arrays were large. Training is sequential as a result.
- Timing on larger datasets and GPU use have not been benchmarked. The device used is recorded in `training_log.json`.
