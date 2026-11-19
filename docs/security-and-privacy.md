# Security and privacy

Synapse is built for health and biomedical data, which is sensitive. This page lists what the tool does, what it does not, and what you are responsible for.

## API keys

- Configs hold the **name** of an environment variable (`api_key_env`), never the key.
- A value that looks like a pasted key is rejected and not echoed.
- Keys are not written to `run_log.json`, validation reports, output tables or generated scripts.
- If a key was ever committed, rotate it.
- Do not paste keys into chat with an AI assistant.

## Code written by a model

In `write_generator` mode the model writes Python that Synapse executes.

- It runs in a subprocess with a timeout (`script_timeout`) and with secret-looking environment variables (names containing KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL) removed.
- It is **not a sandbox**: no filesystem or network isolation. Treat generated code as untrusted and run Synapse inside a container or VM for anything beyond local experiments.
- The accepted script is saved as `generator.py` so you can review it before reusing it.

## Sending real data to a model provider

Few-shot context (`few_shot:`) is sent to the LLM provider.

- `stats` mode sends aggregates (still some information about the real data).
- `rows` mode sends real example rows and requires `acknowledge_external_data: true` for hosted models. Only set it if you are allowed to share those rows.
- A model on `localhost` needs no flag and keeps data on your machine; prefer it for sensitive data.
- Use `exclude_columns` to drop direct identifiers (names, record numbers, addresses, dates of birth, emails, phone numbers).
- Never send real patient records, genomic data of individuals or GPS tracks to a hosted model without the approvals your institution requires.

## Models trained on real data

- Saved models (`.pkl`) are pickles. **Loading a pickle can run arbitrary code**; load only models you trained yourself, never ones from untrusted sources.
- A generative model trained on real patient data can memorise and leak training records. The evaluator's privacy score (distance to closest record, exact-copy rate) is a heuristic check, not a guarantee. It is not differential privacy and not a membership-inference test.
- Keep `data/real/` out of version control (it is git-ignored). Do not commit real data, models trained on it, or `holdout` copies.
- `train.py` deletes held-out copies of real rows after scoring; `schema.json` stores aggregate distributions and key formats, not rows.

## Synthetic does not mean anonymous or valid

- Describe outputs as synthetic. Do not imply they are real or de-identified real data.
- Synthetic health data produced from clinical rules of thumb is not clinically valid and must not inform care, dosing, diagnosis or biological conclusions.
- Generated identifiers, names and emails are fake (`@example.com`). Do not reuse real MRNs, names or addresses.
- GPS-like fields in wearable data can reveal home and routine if copied from real tracks; use fake routes.

## Reporting a vulnerability

Open a private security advisory on the repository, or contact the maintainer, rather than a public issue.
