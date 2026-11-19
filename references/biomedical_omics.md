# Domain profile: Biomedical data (gene-level, RNA-seq)

Gene expression experiments: a count matrix (genes x samples) plus sample metadata and gene annotation. Mostly bulk RNA-seq; notes for single-cell at the end.

> Synthetic data only. It mimics statistical properties of sequencing counts; it carries no real biology and must not be used for biological conclusions.

## Entities
| Table | Grain | Key columns |
|---|---|---|
| `counts` | gene x sample matrix | gene_id (rows), one integer column per sample_id (raw counts) |
| `samples` | one per sample | sample_id, condition (e.g. control/treated or disease/healthy), batch, sex, age, tissue, library_size, rin (RNA integrity), subject_id |
| `genes` | one per gene | gene_id (Ensembl-style, e.g. ENSG00000000001), symbol, biotype (protein_coding, lncRNA, ...), length_bp, gc_content, chromosome |
| `truth` | one per gene (ground truth) | gene_id, is_de, true_log2fc, base_mean, module_id |
| `de_results` (optional) | one per gene | gene_id, base_mean, log2FoldChange, pvalue, padj |
| `tpm` / `logcpm` (optional) | derived matrices | computed from `counts`, never sampled independently |

Provide the matrix in two forms when useful: wide (genes as rows) and tidy long (gene_id, sample_id, count). Always save the `truth` table: it plays the role of `is_anomaly` and lets downstream tools be scored.

## Realism features
- **Counts are negative binomial, not normal.** For gene g in sample j: `count ~ NB(mean = mu_gj, dispersion = phi_g)`, `mu_gj = s_j * base_g * fold_effect(g, condition) * batch_effect(g, batch)`.
- **Gene base means are heavy-tailed:** lognormal across genes (a few very high, most low). Many genes have near-zero counts (often 30-50% of annotated genes are lowly expressed).
- **Mean-variance relationship:** variance exceeds the mean (overdispersion). Use a dispersion trend that decreases with mean, e.g. `phi = a + b / mu` (typical a about 0.05-0.2, b about 0.5-2), plus gene-wise scatter (lognormal).
- **Library sizes vary per sample:** lognormal around 15-40M reads (bulk), with a size factor `s_j` multiplying all genes. Counts therefore do not sum to the same value across samples.
- **Differential expression:** 5-15% of genes are truly DE. Draw `log2FC` from a normal (SD about 1) or a mixture of small and large effects, split up/down. Effect size should be independent of base mean, with low-count genes harder to detect.
- **Gene-gene correlation:** genes are co-expressed in modules (pathways). Generate a few latent factors per sample and load genes onto them (module structure). Without this, PCA and clustering look unrealistically flat.
- **Housekeeping genes:** a small set with stable, high expression and low variance across conditions.
- **Batch effects:** multiplicative gene-specific shifts per batch (log-scale normal, SD about 0.2-0.5), plus batch-linked library size differences. Make batch partly confounded with condition only when asked (it is a common benchmark challenge).
- **Technical bias:** longer genes yield more reads (count proportional to length for the same molar abundance); GC bias is optional. Needed if you also output TPM.
- **Biological replicates differ:** within a condition, samples share the condition mean but vary by dispersion; add subject effects for paired designs.
- **Sample metadata correlated with expression:** sex-linked genes (XIST high in female, Y-genes high in male), age-associated genes with small slopes.
- **Outliers and quality:** an occasional low-RIN or low-depth sample (flag it); rare sample swap or contamination as a labelled anomaly.

## Sanity rules
- Counts are non-negative integers; no NaN in the raw matrix; gene_id unique; sample_id in `counts` columns equals `samples.sample_id`.
- Column sums approximately equal `samples.library_size` (within the multinomial/NB noise you modelled).
- Observed variance is greater than the mean for most expressed genes; the mean-variance plot shows the intended trend.
- TPM columns sum to 1e6 per sample; logCPM is derived from counts and library size.
- Null (non-DE) genes give roughly uniform p-values in a DE test; true DE genes enrich at small p-values; the share of `is_de` recovered at FDR 5% is plausible (neither 0% nor 100%).
- PCA of log-normalized counts separates samples by condition (and by batch if modelled); PC1 should not be library size alone.
- Sex-linked genes agree with `samples.sex`.

## Typical volumes
- Bulk: 15k-60k genes x 6-500 samples (small experiments 3-6 per group; cohort studies hundreds).
- Single-cell: 2k-30k genes x 1k-100k cells, mostly zeros (see below).
- Demo default: 5,000 genes x 12 samples (2 conditions x 2 batches, 3 replicates).

## Choosing a backend and evaluation notes
- **Use a parametric simulation (rules backend or an LLM-written generator), not a GAN.** CTGAN and TVAE are designed for tabular records with few columns; with far more genes than samples (p much larger than n) they overfit or collapse. If you want a conventional learned baseline, model a low-dimensional summary (per-gene mean/dispersion, PCA scores) instead.
- Dedicated simulators exist in the R/Bioconductor ecosystem (Splatter for single-cell, compcodeR and polyester for bulk RNA-seq); verify current names and maintenance before citing them.
- **Evaluation needs different metrics than the default tabular ones.** The built-in evaluator treats rows as records; for a gene x sample matrix transpose to samples x features or evaluate summaries: per-gene mean and variance distributions, dispersion trend, library size distribution, sample-sample correlation, PCA structure, zero fraction, and DE recovery against `truth`. These are not implemented in `evaluation/` yet.
- Real data, if used for few-shot or training, can be human-subject data with consent and access restrictions. Do not send it to a hosted LLM; share only aggregate summaries and prefer a local model.

## Single-cell notes
- Much sparser: 80-95% zeros; model with NB plus dropout (zero-inflation) or a Gamma-Poisson with cell-specific size factors.
- Cell types: a mixture of clusters with marker genes; add doublets and ambient RNA as labelled anomalies.
- Output sparse format (matrix market, h5ad) rather than dense CSV for anything large.

## Pitfalls
- Normal or uniform noise instead of NB (variance structure is wrong, DE tools misbehave).
- Independent genes (no modules), identical library sizes, or no zeros.
- Mixing raw counts, TPM and log values in the same file without labels.
- Reporting "realistic" without showing the mean-variance and PCA checks.
- Real gene symbols with fabricated functions: label annotations as synthetic and arbitrary.
