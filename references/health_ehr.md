# Domain profile: Health data (EHR and disease classification)

Two shapes of data share this profile:
- **Relational EHR:** patients, encounters, diagnoses, labs, vitals, medications, procedures.
- **Flat classification tables:** one row per patient with clinical features and a disease label (Pima-style diabetes, heart disease, CKD, stroke, breast cancer).

> All data is synthetic. Do not present it as real patients, clinically valid, or usable for medical decisions. Thresholds below are well-known reference values used only to make the data plausible.

## Entities (relational EHR)
| Table | Grain | Key columns |
|---|---|---|
| `patients` | one per person | patient_id, birth_date, sex, ethnicity (optional), smoking_status, death_date (nullable) |
| `encounters` | one per visit/admission | encounter_id, patient_id, type (outpatient/inpatient/ED), admit_ts, discharge_ts, department |
| `diagnoses` | one per coded diagnosis | diagnosis_id, encounter_id, icd10_code, is_primary, diagnosed_date |
| `vitals` | one per measurement set | vital_id, encounter_id, ts, sbp, dbp, heart_rate, temp_c, spo2, weight_kg, height_cm, bmi |
| `labs` | one per result | lab_id, encounter_id, ts, loinc_code, test_name, value, unit, ref_low, ref_high, abnormal_flag |
| `medications` | one per order | med_id, encounter_id, drug_name, atc_class, dose, start_date, end_date |

## Flat classification table (diabetes-style)
Typical columns: age, sex, pregnancies (female only), glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree (family history score), hba1c, cholesterol, smoker, and `outcome` (0/1) or a multi-class disease label.

## Realism features
- **Latent risk drives everything.** Draw a per-patient latent `risk` (from age, BMI, family history, lifestyle). Let glucose, HbA1c, blood pressure, lipids and the disease label all depend on it. Never sample label and features independently.
- **Label from features with noise:** `P(diabetes) = sigmoid(a + b*age + c*bmi + d*glucose + family_history)`, then draw the label. Set intercept so overall prevalence matches the target (general adult population roughly 8-15%; Pima-style benchmark sets about 35%).
- **Class imbalance:** keep a realistic minority rate; state the rate in the hand-off.
- **Comorbidity clusters:** diabetes, hypertension, obesity, dyslipidemia and chronic kidney disease co-occur. Sample diagnoses from a joint model (shared latent risk, or conditional probabilities), not independent coin flips.
- **Distributions:** age roughly 18-90 skewed older for inpatient data; BMI lognormal-ish (mean about 27-30, right tail); glucose and HbA1c right-skewed, with the diseased group shifted up; blood pressure correlated with age and BMI; cholesterol approximately normal, triglycerides lognormal.
- **Reference thresholds for labels (adult):** diabetes if HbA1c at least 6.5% or fasting glucose at least 126 mg/dL; prediabetes HbA1c 5.7-6.4% or fasting glucose 100-125 mg/dL; hypertension if SBP at least 130 or DBP at least 80 in recent guidelines (older: 140/90). Use as soft boundaries with measurement noise.
- **Longitudinal behaviour:** visits are irregular (gaps from days to years); chronic patients have more visits; HbA1c drifts slowly with treatment; weight changes gradually; acute events create bursts of encounters, labs and vitals.
- **Missingness is informative (MNAR):** labs are ordered when clinically suspected, so sicker patients have more tests. Missing values are not uniform random. Add a `lab_ordered` mechanism tied to symptoms or diagnoses.
- **Measurement noise and rounding:** vitals rounded to integers, labs to realistic precision, repeated measures vary a little (within-person SD).
- **Units:** keep one unit per test (mg/dL vs mmol/L); if you mix, include the unit column and make values consistent.
- **Seasonality:** respiratory diagnoses peak in winter; ED visits peak evenings and weekends.
- **Anomalies (flag with `is_anomaly`):** impossible-but-plausible entry errors (weight in lb entered as kg, swapped SBP/DBP), duplicated encounters, extreme labs.

## Sanity rules
- `birth_date < admit_ts <= discharge_ts`; `death_date` after the last encounter; no encounters after death.
- Vital ranges: SBP 70-250, DBP 40-150, SBP > DBP, heart rate 30-220, temperature 34-42 C, SpO2 70-100, BMI 12-70.
- `bmi` equals weight_kg / (height_cm/100)^2; recompute it, do not sample it. Same for any derived index (eGFR from creatinine, age and sex).
- Pregnancies and pregnancy diagnoses only for female patients of reproductive age; male-specific diagnoses only for male patients.
- ICD-10 and LOINC codes must have valid formats (letter + digits for ICD-10). Prefer a small curated code list over inventing codes.
- `abnormal_flag` agrees with value vs `ref_low`/`ref_high`.
- In classification tables, avoid physiologic zeros as missing markers (a real Pima-style quirk: glucose = 0). Use NaN, and say so.
- Medication start precedes end; drug class matches indication where modelled (e.g. metformin only with diabetes).

## Typical volumes
- Flat classification: 500-100k rows (Pima-style: 768).
- Relational demo: 1k-10k patients, about 5-10 encounters per patient, 20-60 lab results per patient.

## Choosing a backend and evaluation notes
- **Flat tables with real data available:** CTGAN or TVAE work well. Set `utility` to the label column; TSTR is the key metric. Watch class imbalance (the evaluator uses AUC for binary labels).
- **Relational EHR:** use the rules or LLM write-generator route for temporal logic and clinical rules; GANs do not preserve dates, ordering or arithmetic. Keys and derived columns are handled by the training workflow (`primary_keys`, `foreign_keys`, `derived`).
- **Privacy matters most here.** If a model was trained on real patient data, run the privacy metrics (distance to closest record, exact-copy rate) and treat the result as a heuristic, not a guarantee. Never send real rows to a hosted LLM in few-shot `rows` mode without approval.
- Shift or synthesize all dates; never reuse real MRNs, names or addresses.

## Pitfalls
- Independent sampling of label and features (the classifier then learns nothing real).
- Perfectly clean data; real EHR has missing, inconsistent and duplicated entries.
- Using real dosing, diagnosis or treatment rules as authoritative.
- Sample sizes too small to support the imbalance you asked for (1% prevalence in 500 rows is about 5 cases).
