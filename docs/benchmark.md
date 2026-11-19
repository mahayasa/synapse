# Benchmark plan

A research design for comparing synthetic-data generators with this harness. This is a plan, not a result: no benchmark has been run yet.

## Question

Which kind of generator produces the best synthetic health and biomedical data, and does an agent built on top of a language model work reliably?

## Arms

| Arm | Approach | Synapse backend |
|---|---|---|
| A | LLM writes rows directly | `anthropic_llm` / `openai_compat` with `mode: direct_rows` |
| B | LLM writes a generator (the agent) | same backends with `mode: write_generator` |
| C | Statistical and ML generators (Gaussian copula, CTGAN, TVAE; later diffusion) | `sdv` |
| D | Rule-based baselines | `rules` |
| E | Ablations of B | no domain profile, no validator, no few-shot reference |

Arms C and D need real data to fit (C) or hand-built rules (D). Run one setting with real data available and one without.

## Reference data

Realism is only measurable against real data. Use public datasets with a real hold-out split and check each licence and access terms. Suggested starting points are public tabular datasets with known schemas; for clinical data use only data you are permitted to use, and prefer de-identified public sets.

## Metrics

Use the evaluator ([evaluation.md](evaluation.md)): fidelity, utility (TSTR), privacy, coverage and validity, integrity. Also record cost, time and maximum feasible scale, and for the agent: task success rate, repair iterations, reproducibility and prompt sensitivity.

## Models

Pin exact model ids and dates. A suggested set: three frontier models from different providers, two small models, one open-weight model. Compare reasoning modes as a separate factor. Verify current model names with each provider before starting.

| Tier | Examples |
|---|---|
| Claude | `claude-fable-5-2`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` |
| Other frontier | Current OpenAI and Google flagship models (verify names) |
| Small | Haiku, plus a Flash-class model |
| Open-weight | One Llama, Qwen, DeepSeek or Mistral model |

## Protocol

- 3 to 5 domains and several sizes.
- At least 5 seeds per arm; report confidence intervals and significance tests.
- Same harness, prompts, temperature and settings for every model; log everything (`run_log.json`).
- Equal tuning effort per arm; normalise results by cost.
- Pre-register metrics before running.
- Human or LLM-judge ratings only as a secondary check.

## Threats to validity

- Unequal prompt-tuning effort between arms.
- Public datasets may appear in language-model training data.
- Fidelity metrics can be gamed by copying real data, so always report privacy.
- Evaluation gaps: matrices and time series are not fully covered yet ([roadmap.md](roadmap.md)).

## Hypotheses

- C wins on fidelity to a real flat table.
- B wins on relational integrity, scale and cost.
- A degrades beyond a few thousand rows.

These are hypotheses to test, not findings.
