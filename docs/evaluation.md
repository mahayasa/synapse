# Evaluation

Quality can only be measured against **real data**. The evaluator compares a folder of synthetic CSVs with a folder of real CSVs (matching file names = tables). All scores are between 0 and 1, higher is better.

```bash
python evaluate.py --real data/real --synthetic out/run_a --utility patients:outcome:classification
python evaluate.py --real data/real --synthetic out/a out/b out/c --labels claude gpt ctgan   # leaderboard
```

Options: `--exclude` (extra columns to ignore), `--utility table:target[:classification|regression]`, `--validation-config <run config>` (integrity checks), `--seed`. Reports are written to `evaluation_report.md` and `.json` in each synthetic folder. In a run config, add an `evaluation:` block (`real_data`, `exclude_columns`, `utility`, `seed`, `max_rows`).

## Preparation

Identifier columns are ignored automatically: columns named `id` or ending `_id`, listed in `exclude_columns`, or categorical columns that are more than 90% unique per row. Dates become numeric seconds; integers with 10 or fewer distinct values and booleans are treated as categorical; everything else is numeric or categorical by type.

## Metrics

### Fidelity

| Metric | Definition | Score |
|---|---|---|
| Column shapes | Numeric: 1 minus the Kolmogorov-Smirnov statistic between real and synthetic. Categorical: 1 minus total variation distance | Mean over columns |
| Pair trends | Numeric pairs: 1 minus half the gap between Spearman correlations. Other pairs: 1 minus total variation distance between joint distributions (numeric columns binned into quintiles) | Mean over column pairs (first 25 columns) |
| C2ST | A gradient-boosting classifier tries to tell real from synthetic rows (5-fold cross-validated AUC). `score = 1 - max(0, 2 * (AUC - 0.5))` | 1 means indistinguishable, 0 means trivially separable |

### Utility

TSTR (train on synthetic, test on real). A model is trained on the synthetic table and on 70% of the real table, and both are tested on the other 30% of the real table. The score is synthetic performance divided by real performance, clipped to 0..1. Performance is Gini (2 x AUC - 1) for binary targets, macro-F1 for multiclass, R-squared for regression. Skipped (n/a) if the real baseline is not clearly above chance.

### Privacy

Distance to the closest record (DCR) and exact-copy rate, each compared with a real holdout. Real rows are split 80/20; synthetic rows and the 20% holdout are both matched to their nearest neighbour in the 80% training part.

`score = min(1, median DCR synthetic / median DCR holdout) x (1 - max(0, exact-copy rate synthetic - exact-copy rate holdout))`

Synthetic rows that are no closer to training rows than unseen real rows are score about 1; copied rows score near 0. This is a heuristic, not a formal guarantee such as differential privacy, and it does not replace a membership-inference test.

### Coverage and validity

Boundary adherence (share of synthetic values inside the real min-max), range coverage, category coverage, and novel-category rate.

### Integrity

Share of the run's `validation` checks (primary keys, foreign keys, ranges, not-null) that passed.

## Composite score

A heuristic weighted mean of the available groups: fidelity 0.40, utility 0.20, privacy 0.15, integrity 0.15, coverage and validity 0.10 (weights in `evaluation/report.py`, renormalised over the groups that could be computed). **Always report the groups and the weakest columns, not only the composite.** The report lists the weakest columns and column pairs per table.

## Caveats

- Scores are noisy with fewer than about 500 real rows; the report warns about this.
- A distribution-level metric cannot prove data is realistic for a use it does not cover.
- The evaluator treats rows as independent records. For gene-by-sample matrices or time series it can only evaluate a tabular view; metrics such as mean-variance trend, PCA structure, differential-expression recovery, autocorrelation and sequence distance are not implemented yet (see [roadmap.md](roadmap.md)).
- A detector AUC slightly below 0.5 can occur by chance on small samples.
- Validated on constructed cases (faithful, distorted, exact copy); not yet validated on real public datasets.
