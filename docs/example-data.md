# Example data (few-shot)

You can show Synapse what your data looks like and have the generated data follow it. This is **few-shot guidance, not training**: the example steers the generator that is written; no model weights change. For models that really learn from your data (CTGAN, TVAE, Gaussian copula) see [training.md](training.md).

## Ways to give example data

| Where you work | How |
|---|---|
| **Claude session** (`/synapse`) | Paste a table into the chat, or give the path of a CSV, TSV or XLSX file: "generate data like `data/real/my_patients.xlsx`". Claude saves pasted text under `data/real/`, profiles it, and follows the rules below |
| **Command line** | `python profile_data.py <file-or-folder>` prints the profile; `--export` writes normalised CSVs with an optional holdout |
| **API backends** (`anthropic_llm`, `openai_llm`, `openai_compat`) | A `few_shot:` block in the config. `real_data` can be a folder or a single file; see [backends.md](backends.md) |

## Supported formats

| Input | Result |
|---|---|
| `.csv`, `.tsv` | One table named after the file |
| Pasted text saved as `.txt` or `.md` | One table. The delimiter is detected: markdown tables, tab, comma, semicolon, pipe, or aligned columns |
| `.xlsx`, `.xlsm` (needs `openpyxl`) | One table per non-empty sheet, named after the sheet. A workbook with one default-named sheet ("Sheet1") uses the file name |
| A folder | Every `.csv`, `.tsv`, `.xlsx` and `.xlsm` file in it (`.txt`/`.md` only when named directly) |
| `.xls` (old Excel) | Not supported: save as `.xlsx` or `.csv` |

Multiple tables (sheets or files) are treated as a relational dataset: shared key columns are reported as likely foreign keys.

## What the profile contains

`python profile_data.py data/real/my_patients.xlsx --exclude name email` prints, per table:

- **Columns:** type, range and quartiles for numbers, category mixes for text and for flags or small integer codes (so class balance such as `outcome: 0=83%, 1=17%` is visible), missing-value rates, and a masked shape for identifier-like text (for example `A99999`, never the real value).
- **Detected:** likely keys (unique, never empty), derived columns (a column that is the product or sum of two others, such as `total = qty * unit_price`), and strong correlations (Spearman, absolute value 0.4 or more).
- **Likely foreign keys** between tables.
- **Warnings:** sample size ("very small" under 30 rows, "small" under 100, "modest" under 500), columns that look like personal data by name (name, email, phone, address, ID numbers, date of birth, GPS), mostly-empty columns, constant columns, duplicated rows.
- **Guidance** on how to use it.

Options:

| Option | Effect |
|---|---|
| `--exclude col ...` | Drop columns before profiling and exporting (use for identifiers) |
| `--mode rows --rows N` | Also print N example rows per table. Default is `stats`: summaries only |
| `--export DIR` | Write normalised CSVs to `DIR`, one per table |
| `--holdout F` | With `--export`: write `DIR/train` and `DIR/holdout` (F is the held-out fraction, for example 0.3). Tables under 100 rows are not split. The profile is computed on the train part only |
| `--seed N` | Seed for the split and for example rows |
| `--json` | Print the profile as JSON |

The name-based personal-data check is a heuristic: it can flag harmless columns such as `drug_name` and can miss unusual names. Always check the column list yourself.

## Rules the agent follows (in a Claude session)

1. **Privacy first.** The data is now in the chat and is sent to Anthropic. Claude tells you, asks you to confirm it is allowed, and offers to drop identifiers first. It does not continue with identifiable records without your confirmation.
2. **Few-shot, not copying.** The sample sets column names, types, formats, ranges, category mixes, missingness, correlations, keys and derived columns. Sample rows and real identifiers are never hard-coded into the generator or the output.
3. **The sample wins over the built-in domain profile.** Differences (columns, units, prevalence, ranges) are listed in the hand-off.
4. **Honest about size.** Under about 30 rows only formats, ranges and category names are supported. A few hundred rows are needed for a meaningful evaluation.
5. **Evaluate against held-out rows** when there are about 200 rows or more.

## Holdout evaluation workflow

```bash
python profile_data.py data/real/my_patients.xlsx --exclude name email \
    --export data/real/prepared --holdout 0.3          # profiles the train part only
# ... generate data using the train profile only ...
python evaluate.py --real data/real/prepared/holdout --synthetic out/run_a
```

The generator never sees the holdout rows, so the evaluation measures how well the output matches data it was not shown. With too little data for a holdout, evaluate against the whole sample (`--real data/real/prepared`) and treat the result as a consistency check, not a held-out test: the generator saw those rows.

## For API backends

```yaml
few_shot:
  real_data: data/real/my_patients.xlsx    # a file or a folder
  mode: stats                              # stats | rows
  exclude_columns: [name, email]
domain_profile: health_ehr                 # optional; the real-data reference wins if both are set
```

`rows` mode sends real example rows to the model provider and requires `acknowledge_external_data: true` unless the model is local. If both `domain_profile` and `few_shot` are set, the prompt tells the model the real-data reference wins where they disagree and records the differences in a comment, and `run.py` prints a warning. See [backends.md](backends.md) and [security-and-privacy.md](security-and-privacy.md).

## Limitations

- A small sample teaches formats and rough distributions, not exact relationships. The profile only reports pairwise relationships it can see.
- Few-shot guidance does not guarantee a statistical match; use a trained model if you have a few hundred rows or more and need one.
- The agent can still reproduce a sample row by chance, or deliberately by mistake. The evaluator's exact-copy and distance checks help, but only when there is enough data.
- Matching a sample's statistics does not make data clinically or biologically valid.
- Tested offline with constructed samples; not yet with real datasets.

## Troubleshooting

| Message | Fix |
|---|---|
| `reading .xlsx needs openpyxl` | `pip install openpyxl` |
| `legacy .xls is not supported` | Save the file as `.xlsx` or `.csv` |
| `could not detect columns` | Pasted text needs a header row and consistent delimiters (or a markdown table) |
| `--holdout needs --export DIR` | Add `--export <folder>` |
| `few_shot.mode 'rows' sends real rows ...` | Set `acknowledge_external_data: true`, use a local model, or use `stats` mode |
