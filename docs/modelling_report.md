# Modelling report: breakfast iAUC prediction (M4)

This report holds the methods, commands, diagnostics and findings of the M4 predictive demonstration. The project [README](../README.md) gives only a summary. For a guided walkthrough, see the [end-to-end demonstration](../demo/README.md#m4-participant-aware-prediction).

These are internal-validation results on a dataset-derived outcome. They make no clinical claims.

## 1. Reference experiment

A small, single-purpose experiment on the M3 analysis table, not a general ML framework. The random forest needs the optional extra (`pip install -e ".[experiment]"`, scikit-learn). Without it, the naive and linear models still run and the report says the forest was skipped.

```bash
.venv/bin/python -m pghd_ready.conditions.cardiometabolic.experiments.breakfast_iauc \
  --config configs/experiments/breakfast_iauc.yaml \
  --task-dir data/processed/cgmacros/tasks/postprandial_breakfast \
  --out data/processed/cgmacros/experiments/breakfast_iauc
```

### Design (configured in `configs/experiments/breakfast_iauc.yaml`)

- **Target:** the signed iAUC from M3, all calculable breakfasts, negatives kept.
- **Predictors:** the approved set only (meal energy and macronutrients, baseline glucose, HbA1c, BMI, age). The code refuses any other column.
- **Validation:** leave-one-participant-out (LOPO), per device. All breakfasts of a participant stay in the same partition. No tuning, no nested CV, no hold-out set.
- **Models:** training-fold mean (naive), ordinary least squares, and a random forest with fixed settings.
- **Primary metric:** participant-macro MAE (MAE per held-out participant, averaged). Pooled MAE, RMSE, R² against naive and calibration are secondary. Intervals come from a bootstrap of whole participants, paired across models.
- **Sensitivity analyses:** without baseline glucose; and the breakfasts calculable on both devices.

### Outputs

`experiment_report.md`, `metrics.json`, `model_metadata.json` and `experiment_manifest.json` (aggregate: input hashes back to M1, config hash, package versions), plus `split_assignments.parquet` and `oof_predictions.parquet`, which are participant-level and must stay local. Reruns are byte-identical.

## 2. Model refinement: collinearity, fasting glucose and ridge

A separate experiment (`configs/experiments/breakfast_iauc_refinement.yaml`, module `experiments/breakfast_iauc_refinement.py`) reads the M4 outputs without changing them. It uses the same rows and folds, and checks that its reference linear model reproduces the M4 predictions exactly. It compares, per device:

- the M4 linear model (A);
- A without total energy (B);
- B plus study-start fasting glucose (C);
- ridge on B and on C (D1, D2).

```bash
.venv/bin/python -m pghd_ready.conditions.cardiometabolic.experiments.breakfast_iauc_refinement \
  --config configs/experiments/breakfast_iauc_refinement.yaml \
  --task-dir data/processed/cgmacros/tasks/postprandial_breakfast --m1 data/processed/cgmacros \
  --reference-dir data/processed/cgmacros/experiments/breakfast_iauc \
  --out data/processed/cgmacros/experiments/breakfast_iauc_refinement
```

- **Ridge:** predictors are standardised with training-fold statistics only. Alpha comes from a fixed grid, chosen by inner leave-one-participant-out over the training participants (scaling is refitted in every inner fold), on participant-macro MAE, with ties going to the larger alpha.
- **Comparisons:** paired improvement = MAE(reference) − MAE(candidate), using the same participant resamples as M4.
- **Diagnostics:** the report adds variance inflation factors and coefficient stability across folds. Coefficients are descriptive, not independent or clinical effects.
- **Fasting glucose** (`fasting_glucose_lab`) is one laboratory value per participant, documented as fasting and taken at study start. Its fasting status is documented but not independently verifiable. Its collection date and timing relative to individual breakfasts cannot be verified. It is distinct from the per-breakfast baseline glucose, and it is strongly correlated with HbA1c.

## 3. Findings

On CGMacros (45 participants; 422 Libre and 392 Dexcom calculable breakfasts, analysed separately), leave-one-participant-out linear regression reduced participant-macro MAE by about a quarter to a third compared with a naive mean.

The preferred specification is **Model B**, linear regression without total energy. It is no more accurate than the original nine-predictor model, but it removes severe collinearity. Fasting glucose and ridge regression added no detectable value.

## 4. Limitations

- The outcome is a dataset-derived quantity (calculable iAUC from CGMacros readings as released), not a measured physiological glucose response. See the M3 limitations in the [README](../README.md#preparing-an-analysis-table-m3).
- CGMacros CGM values were interpolated by the dataset authors before release, and the baseline may contain post-meal information.
- Validation is internal only (one dataset, no external hold-out), with no hyperparameter tuning outside the ridge inner loop.
- Devices are analysed separately and not compared.
- Coefficients are descriptive and carry no clinical interpretation.

Further reasoning behind the decision is in the [end-to-end demonstration](../demo/README.md#m4-participant-aware-prediction).
