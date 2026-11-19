# Example: diabetes classification table

A small synthetic Pima-style diabetes classification table produced by following `references/health_ehr.md`.

```bash
python generate_diabetes.py --seed 42 --rows 500 --out data
```

`data/diabetes.csv` (500 rows, seed 42, about 15% positive) is included. All values are synthetic; nothing here is real patient data and it is not clinically valid.

## Columns

`patient_id`, `age`, `sex`, `pregnancies`, `bmi`, `family_history` (0-1 score), `smoker`, `sbp`, `dbp`, `fasting_glucose`, `hba1c`, `triglycerides`, `outcome` (1 = diabetes), `is_anomaly`, `anomaly_type`.

## Design

- A latent risk (age, BMI, family history, smoking) sets the probability of diabetes; the label is drawn from it, with the intercept calibrated to the target prevalence.
- Glucose, HbA1c, blood pressure and triglycerides are generated conditional on the label with deliberate overlap: some treated diabetics have near-normal glucose and some non-diabetics have raised glucose. A classifier reaches an AUC of about 0.91 (cross-validated), not 1.0.
- HbA1c is missing in about 30% of rows, more often for low-glucose patients (informative missingness).
- 2% of rows carry flagged entry errors: BMI unit errors and swapped systolic/diastolic pressure. They stay within plausible ranges and are marked in `is_anomaly` and `anomaly_type`.
- Pregnancies are zero for males; systolic pressure exceeds diastolic outside flagged anomalies.
- Reproducible: the same seed gives a byte-identical file.

## Validate

```bash
python ../../scripts/validate.py --dir data --pk diabetes=patient_id \
  --range diabetes.age=18:100 diabetes.bmi=12:70 diabetes.sbp=40:250 diabetes.dbp=40:150 \
          diabetes.fasting_glucose=40:500 diabetes.hba1c=3:16 \
  --notnull diabetes.patient_id diabetes.outcome
```

Fidelity was **not** measured: there was no real dataset to compare with. To score a synthetic table against real data, use `evaluate.py` (see `docs/evaluation.md`).
