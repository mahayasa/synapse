# Domain profiles

A domain profile is a markdown file in `references/` that tells the agent what realistic data looks like in a field, which rules to check, and which backend and metrics suit it. LLM backends paste the profile (`domain_profile: <name>`) into the prompt; in a Claude session the skill loads it at step 2 of the process.

> Physiological, clinical and biological values in the profiles are rules of thumb used to make synthetic data plausible. They have **not** been checked against primary sources. Have a domain expert review them before relying on them, and never present the output as clinically or biologically valid.

## Included profiles

| Profile | Covers | Suited backend |
|---|---|---|
| [`health_ehr`](../references/health_ehr.md) | Relational EHR (patients, encounters, diagnoses, vitals, labs, medications) and flat disease-classification tables (diabetes-style): latent risk, comorbidity clusters, informative missingness, label drawn from features with noise, clinical sanity rules | Flat tables with real data: CTGAN or TVAE. Relational and temporal logic: rules or an LLM-written generator |
| [`biomedical_omics`](../references/biomedical_omics.md) | Gene-level RNA-seq: negative-binomial counts with overdispersion, library sizes, gene modules, batch effects, differential-expression ground truth, single-cell notes | Parametric simulation (rules or LLM-written generator). GANs are a poor fit when genes far outnumber samples |
| [`exercise_wearables`](../references/exercise_wearables.md) | Smartwatch and sport data: heart-rate dynamics, workouts, sensor time series, daily and sleep summaries, sensor artefacts, GPS privacy | Rules or an LLM-written generator for sequences; CTGAN for tabular summaries |

Each profile has the same sections: Entities, Realism features, Sanity rules, Typical volumes, Choosing a backend and evaluation notes, Pitfalls.

## Profiles versus real data

| Path | Uses the profile? | Uses real data? | If they disagree |
|---|---|---|---|
| `sdv` (CTGAN, TVAE, copula) | No | Yes, learns only from it | Real data wins completely |
| `rules` | No | No | Profile ignored; fixed script output |
| LLM with `domain_profile` only | Yes | No | Follows the profile; may not match your data |
| LLM with `few_shot` only | No | Yes | Matches your data |
| LLM with both | Yes | Yes | Risky: no precedence rule today, the model may blend them |
| Evaluator | No | Yes | A mismatch shows up as low fidelity, not an error |

The sanity rules in a profile are guidance text for the model and for Claude; they are not enforced by code. Only the config's `validation` checks and the evaluator actually run.

Practical advice: if your real data differs from the profile (columns, units, prevalence), use `sdv`, or an LLM with `few_shot` and no `domain_profile`.

## Adding a profile

1. Copy the section headings of `references/health_ehr.md`.
2. Fill in realistic entities, behaviours (with named parameters), sanity rules and typical volumes.
3. Say which backend suits the domain and which evaluation metrics apply, including gaps.
4. Cite sources for clinical or physiological constants.
5. Use it with `domain_profile: <file name without .md>`.

Sensitive domains: never send real patient, genomic or location data to a hosted model, and describe results as synthetic.
