# PGHD-Ready

**An open-source research toolkit for integrating, assessing, and establishing the predictive fitness of patient-generated health data (PGHD).**

## Overview

Patient-generated health data (PGHD) collected through digital health applications, wearable devices, connected sensors, self-tracking tools, and patient-reported outcomes offer valuable opportunities to identify emerging health risks and strengthen real-world evidence for personalized healthcare.

However, these data are often fragmented across sources, inconsistent in structure, variable in quality, and difficult to align over time. These challenges can affect the reliability of downstream statistical analyses and predictive models.

**PGHD-Ready** aims to address these challenges by providing reusable methods and components for integrating heterogeneous PGHD, evaluating data quality, and assessing its suitability for predictive research.

Rather than building a separate processing pipeline for every dataset or health condition, the toolkit follows a **configuration-first, modular architecture** designed to accommodate different data sources and research applications while preserving data context, provenance, and clinical meaning.

## Research Focus

The central research focus is understanding how the characteristics and quality of patient-generated health data influence its suitability for identifying predictive clinical signals.

Key areas of investigation include:

- **Data integration:** Harmonizing heterogeneous data sources, schemas, timestamps, measurement units, and clinical concepts.
- **Data quality:** Assessing completeness, missingness, temporal coverage, sampling irregularity, and consistency across modalities.
- **Temporal alignment:** Evaluating how differences in measurement frequency and event timing affect analytical reliability.
- **Predictive fitness:** Developing and evaluating methods to assess how data quality, modality availability, and integration decisions influence the suitability of PGHD for specific predictive tasks.
- **Reproducibility:** Maintaining transparent mappings, transformation records, validation rules, and machine-readable quality assessments.

## Reusable Architecture

PGHD-Ready is designed around three complementary components:

1. **Shared core:** Reusable ingestion, schema mapping, standardization, validation, provenance tracking, temporal alignment, and data-quality assessment functionality.
2. **Condition-specific modules:** Research logic, clinical concepts, and analytical methods relevant to particular health domains.
3. **Data source integrations:** Configuration-based mappings for common data formats, with optional custom adapters for sources requiring specialized transformations.

Where possible, new data sources should be integrated through declarative configuration rather than requiring new source-specific Python code. Existing interoperability standards and open-source tools will be leveraged where appropriate.

This architecture is intended to allow researchers to reuse the same core functionality across datasets and health conditions, rather than rebuilding data preparation and quality-assessment workflows for each application.

## PGHD Domains

PGHD-Ready is intended as a cross-condition research framework rather than a toolkit for a single disease. It uses four broad, conceptual PGHD domains:

| Domain | Examples |
|---|---|
| **Physiological** | Continuous and intermittent glucose, heart rate, blood pressure and other vital signs, weight and body composition, biomarkers from connected devices |
| **Behavioural** | Meals and nutrition, physical activity and exercise, sleep, other behaviours captured by digital health apps and self-tracking tools |
| **Medication-related** | Prescribed regimens, patient-reported medication-taking events, missed, delayed or skipped doses, dose timing and reasons for non-adherence |
| **Patient-reported** | Symptoms and side effects, patient-reported outcomes and wellbeing, structured questionnaires, free-text journal entries |

These domains are a way of describing the data, not separate software modules. A single data stream or event can belong to more than one domain. The shared core handles ingestion, standardisation, provenance, temporal alignment and baseline quality assessment in the same way across domains. Clinical concepts, derived measures, prediction tasks and analytical methods live in condition-specific modules.

The project does not attempt to define a universal healthcare data model. New concepts are added through configuration and condition modules when a research question needs them.

## Initial Research Application

The first implementation focuses on cardiometabolic health, using the CGMacros dataset to investigate how data quality, measurement characteristics, temporal alignment, and modality availability influence the reliability and predictive suitability of integrated PGHD.

A differently structured synthetic digital health dataset will subsequently be used to evaluate the portability and reusability of the ingestion architecture.

## Longer-Term Direction

The longer-term ambition is to extend PGHD-Ready across multiple health domains, potentially including behavioral health (for example mood, sleep, symptoms, medication adherence and patient-reported outcomes), while maintaining a shared foundation for integrating and assessing heterogeneous patient-generated data.

A planned future research question is whether patient-reported medication adherence data provides incremental predictive value for cardiometabolic outcomes when combined with physiological and behavioural PGHD. CGMacros does not contain medication or adherence records, so investigating this empirically will require a separate real-world dataset.

By combining reusable data infrastructure with condition-specific research methods, the project aims to support more systematic and reproducible use of PGHD in real-world evidence generation and predictive healthcare research.

## Intended Outputs

The toolkit aims to produce:

- Standardized, analysis-ready datasets.
- Reusable source mappings and configuration templates.
- Machine-readable data-quality profiles.
- Task-specific predictive-fitness assessments.
- Data provenance and transformation records.
- Reproducible analytical reports.

## Project Status

**Early research and development.** This is an exploratory research initiative, not a clinically validated decision-support system.

| Capability | Status |
|---|---|
| Common representation (observations, events, participant attributes, streams) with schema validation | Implemented |
| Explicit annotation of documented issues, suspected anomalies, semantic ambiguities and assumptions | Implemented |
| Three-state sampling origin per observation (native / interpolated / unknown, default unknown) with explicit inclusion policies | Implemented as a generic representation. Row-level classification of CGM values is not planned (deferred research consideration) |
| Shared concept registry plus condition-specific registries, with optional many-to-many PGHD domain tags | Implemented (the four-term domain vocabulary is fixed in code; extending it requires a code change) |
| Configuration-driven ingestion of delimited-text (CSV) sources, with provenance, rejected-record preservation and an analyst-readable ingestion report | Implemented, tested on a small synthetic example |
| CGMacros base ingestion (CGM, Fitbit, meals, participant attributes) with independent reconciliation tests | Implemented, run locally only (data not distributed) |
| Definition-based consistency checks (M2a; first rule: non-HDL = TC − HDL) with reviewed, content-hashed approval and per-dataset run authorisation | Implemented; CGMacros run in review mode only (in-memory, aggregate) |
| Descriptive data-quality profiling (M2b: completeness, observed timestamp coverage, sampling intervals, gaps against documented or configured intervals, sampling origin), per participant, stream and day, with an aggregate report | Implemented, run locally on CGMacros |
| Plausibility rules and cross-stream (Libre vs Dexcom) agreement | Planned |
| Analytical preparation (M3): generic time windows, event alignment, attribute joins and an eligibility ledger with exclusion counts; first task: breakfast glucose response (signed iAUC, per CGM device) | First workflow implemented and run locally on CGMacros (outcomes per device; no modelling) |
| Predictive research demonstration (M4): breakfast iAUC from meal composition, baseline glucose and participant characteristics; naive, linear, random-forest and ridge models; leave-one-participant-out, per device | Complete. Run locally on CGMacros (internal validation only). Preferred specification: reduced linear model without total energy (Model B) |
| JSON, FHIR or open mHealth inputs; free-text sources | Not implemented |
| Medication-related and patient-reported data representations | Not implemented (see roadmap) |

Roadmap (each step is reviewed before the next begins):

1. **M1** Foundation, generic ingestion engine and CGMacros ingestion. *(Complete.)*
2. **M2** Standalone data-quality assessment. *(First milestones complete: consistency checks (M2a) and descriptive profiling (M2b). Plausibility rules and cross-stream agreement are still planned.)*
3. **M3** Temporal alignment and task fitness. *(First workflow complete: breakfast glucose-response preparation, run on CGMacros. Further tasks are not planned yet.)*
4. **M4** Bounded predictive research demonstration: breakfast iAUC, per device, leave-one-participant-out. *(Complete. Preferred specification: Model B, reduced linear regression without total energy.)* A second, differently structured synthetic source is now treated as supporting test infrastructure, not a separate research milestone.
5. **M5** Not yet defined.
6. **M6** Medication adherence and multimodal PGHD: a medication event representation, prescribed regimens kept separate from patient-reported taking events, configurable ingestion of medication-tracking exports, and a synthetic adherence example.

## Trying the ingestion engine

A small, entirely fictional example dataset is included in `data/synthetic/wearable_demo/`. Its README documents each intentional data problem.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pghd-ready ingest \
  --source    data/synthetic/wearable_demo/config/source.yaml \
  --concepts  configs/concepts/shared.yaml configs/conditions/cardiometabolic.yaml \
  --data-root data/synthetic/wearable_demo/raw \
  --out       data/processed/wearable_demo
```

The run writes standardised Parquet tables, detailed audit files and `ingestion_report.md`. The report explains in plain language what was read, mapped, transformed, flagged or rejected. A successful ingestion run does not mean the data are fit for any particular analysis.

## Profiling data quality (M2b)

`pghd-ready profile` describes a finished ingestion run. It does not modify, interpolate, resample or impute observations, and it produces no findings or thresholds.

```bash
.venv/bin/pghd-ready profile --m1 data/processed/cgmacros --out data/processed/cgmacros/quality
# optional, for a stream with no documented interval:  --expected-interval fitbit_hr=60   (and --gap-factor, default 1.5)
```

Outputs:
- `profile_report.md` (aggregate only) and `profile_manifest.json` (input and output hashes, code commit, parameters).
- Participant-level tables, which must stay local: `stream_profile`, `daily_profile` and `gaps`, plus `dataset_summary` (one row per stream).

Metric definitions are in `src/pghd_ready/core/profiling.py`. Key limitations:
- **Observed span is not wear-time.** It is the time between the first and last observed timestamp, and readings do not establish that a device was worn.
- **Gaps are assessed only where an expected interval exists.** That interval must be documented in the source configuration or passed explicitly with `--expected-interval`. It is never inferred from the data. For CGMacros, only Libre (900 s) and Dexcom (300 s) have documented intervals, so Fitbit gaps are reported as not assessed.
- **CGMacros CGM values were interpolated by the dataset authors to 1-minute rows.** For CGM, use slot coverage at the documented interval rather than counts of 1-minute rows. Every sampling origin is `unknown`.
- **Dates are as recorded.** CGMacros dates are shifted, so calendar days are not real dates.

## Preparing an analysis table (M3)

`pghd-ready prepare` runs a task definition on a finished ingestion run. It writes:
- a per-candidate eligibility ledger, window readings and an analysis table (all participant-level, so they must stay local);
- an aggregate `task_report.md` and a manifest.

No model is fitted, and observations are never resampled or imputed.

```bash
.venv/bin/pghd-ready prepare --task configs/tasks/postprandial_breakfast.yaml \
  --m1 data/processed/cgmacros --out data/processed/cgmacros/tasks/postprandial_breakfast
```

Generic mechanics are in `core/alignment.py` and `core/eligibility.py`. Research-specific definitions are in the task config and `conditions/cardiometabolic/tasks/postprandial.py`: outcome, baseline, meal separation, predictors and calculability rules.

Limitations of the breakfast workflow:
- **Outcome thresholds are operational, not clinically validated.** Minimum baseline readings are 1 (Libre) and 3 (Dexcom). The longest interval the trapezoid may span is 15 min (Libre) and 5 min (Dexcom). These are based on the documented device intervals.
- **The thresholds apply to the readings as released, not to native device samples.** CGMacros CGM readings are one-minute values that the dataset authors document as interpolated, and their sampling origin is reported as `unknown`. So the thresholds cannot detect gaps that were filled before release, and the iAUC may reflect that interpolation.
- **iAUC is signed** (area below baseline is not clamped) and is reported separately per device. No comparison between devices is made.
- **The outcome is a dataset-derived quantity, not a measured physiological glucose response.** It is computed from interstitial CGM values as released (after the authors' interpolation), relative to a baseline computed from the same values, under operational calculability rules. It should be described as "calculable iAUC from CGMacros readings", not as a participant's true postprandial response.
- **The baseline may contain post-meal information.** If the authors' interpolation is linear between device readings, one-minute values just before the meal lie on a line towards the first device reading at or after the meal: up to the last 15 min of the baseline window for Libre and 5 min for Dexcom. Any later use of the baseline as a predictor must take this into account.
- **Non-calculable breakfasts are not missing at random by construction.** They are excluded for device-data reasons (for example no reading at t0 + 120 min), which differ between devices, so the calculable sets differ by device.

## Breakfast iAUC prediction experiment (M4)

A small, single-purpose experiment on the M3 analysis table, not a general ML framework. The random forest needs the optional extra (`pip install -e ".[experiment]"`, scikit-learn); without it, the naive and linear models still run and the report says the forest was skipped.

```bash
.venv/bin/python -m pghd_ready.conditions.cardiometabolic.experiments.breakfast_iauc \
  --config configs/experiments/breakfast_iauc.yaml \
  --task-dir data/processed/cgmacros/tasks/postprandial_breakfast \
  --out data/processed/cgmacros/experiments/breakfast_iauc
```

Design (configured in `configs/experiments/breakfast_iauc.yaml`):
- **Target:** the signed iAUC from M3, all calculable breakfasts, negatives kept.
- **Predictors:** the approved set only (meal energy and macronutrients, baseline glucose, HbA1c, BMI, age). The code refuses any other column.
- **Validation:** leave-one-participant-out, per device. All breakfasts of a participant stay in the same partition. No tuning, no nested CV, no hold-out set.
- **Models:** training-fold mean (naive), ordinary least squares, and a random forest with fixed settings.
- **Primary metric:** participant-macro MAE (MAE per held-out participant, averaged). Pooled MAE, RMSE, R² against naive and calibration are secondary. Intervals come from a bootstrap of whole participants, paired across models.
- **Sensitivity analyses:** without baseline glucose; and the breakfasts calculable on both devices.

Outputs: `experiment_report.md`, `metrics.json`, `model_metadata.json` and `experiment_manifest.json` (aggregate: input hashes back to M1, config hash, package versions), plus `split_assignments.parquet` and `oof_predictions.parquet`, which are participant-level and must stay local. Reruns are byte-identical.


### Model refinement (M4): collinearity, fasting glucose and ridge

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

### M4 findings (complete)

Cohort: CGMacros, 45 participants. 422 calculable Libre and 392 calculable Dexcom breakfasts, analysed independently and never pooled.

**Results.** Participant-macro MAE in mg/dL × min, with 95% participant-bootstrap intervals; leave-one-participant-out.

| Model | Libre | Dexcom |
|---|---|---|
| Naive (training mean) | 2,987 [2,551, 3,491] | 3,303 [2,794, 3,912] |
| A: linear, 9 approved predictors (reference) | 2,088 [1,730, 2,541] | 2,455 [2,080, 2,893] |
| **B: A without total energy (preferred)** | **2,093 [1,731, 2,547]** | **2,451 [2,075, 2,889]** |
| C: B + study-start fasting glucose | 2,131 [1,745, 2,594] | 2,508 [2,115, 2,966] |
| D1 / D2: ridge on B / C | 2,088 / 2,091 | 2,427 / 2,458 |
| Random forest (original M4, 9 predictors) | 2,331 [1,925, 2,815] | 2,849 [2,413, 3,337] |

**Against the naive baseline.** Linear regression clearly improves on it: macro MAE is lower by 899 [507, 1,317] on Libre and 848 [450, 1,286] on Dexcom. R² relative to naive is about 0.40–0.42, and the calibration slope is 0.94. The random forest was not an improvement over linear regression.

**Preferred specification: Model B.** B is not more accurate than A: the paired difference is −5 [−22, 13] on Libre and 4 [−1, 9] on Dexcom. Its advantage is a more defensible, less collinear specification with no meaningful loss in prediction performance:
- the largest variance inflation factor falls from 59 (Libre) and 156 (Dexcom), both for energy, to about 2;
- its coefficients keep the same signs on both devices, while A's energy coefficient changes sign between devices;
- its macronutrient coefficients vary over a much narrower range across folds.

**Fasting glucose and ridge.** Neither showed a detectable improvement.
- Fasting glucose is strongly correlated with HbA1c (Spearman 0.87).
- Good prediction does not establish individual predictor effects. Collinear predictors can trade weight with little change in predictions, so no coefficient is an independent or clinical effect.

**Limitations:**
- Small cohort: 45 independent participants.
- Internal participant-level validation only, not external validation.
- A designed breakfast protocol (protein shakes of set macronutrient composition), not unrestricted everyday meals.
- Released CGM data are one-minute values interpolated by the dataset authors, which may affect both the baseline and the iAUC.
- The target is a dataset-derived analytical quantity, not a clinically validated outcome.
- No demonstrated clinical utility or generalisability to other populations, devices, conditions or meal settings.

**What M4 demonstrates.** An end-to-end, reproducible workflow built with the toolkit: integrated PGHD (M1), quality assessment (M2), task-specific analytical preparation (M3), and participant-aware validation and modelling (M4). Lineage is hash-linked and outputs are deterministic. M4 is a bounded research demonstration, not a general-purpose ML platform or a validated clinical prediction system.

**Provenance note.** Both experiment manifests currently record code commit `0894ee2-dirty`, because they were run before the experiment code was committed. Rerun both experiments after committing to obtain clean provenance.

## Development and Tests

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt -e ".[dev]"
env -u PYTHONPATH .venv/bin/python -m pytest
```

- The synthetic tests always run. They use only the fictional `data/synthetic/wearable_demo` dataset and data generated inside the tests.
- **Real-data reconciliation tests** (`tests/test_cgmacros_real.py`, marker `real_data`) also run by default, but only where CGMacros is available locally. Without it they are reported as skipped, so the suite passes on any machine.
- These tests compare the ingestion engine with an independent reference profiler (`tests/cgmacros_reference.py`). Their assertions and failure messages are aggregate-only. With the data present they add roughly 40 seconds.
- Run only them with `pytest -m real_data`, or exclude them with `pytest -m "not real_data"`.

**The CGMacros dataset is not included in this repository and must never be committed.** To run the real-data tests or the CGMacros ingestion locally:

1. Download CGMacros v1.0.0 from PhysioNet (https://doi.org/10.13026/3z8q-x658) under its CC BY-NC-SA 4.0 licence, and extract it to `data/raw/cgmacros/` (one `CGMacros-0XX/` folder per participant, plus `bio.csv`).
2. Download the official data dictionaries (`DataDictionary_*.csv`) from the same PhysioNet page into `data/raw/cgmacros_docs/`. The reference profiler reads documented ranges from them.
3. Run the ingestion:
   ```bash
   .venv/bin/pghd-ready ingest --source configs/sources/cgmacros.yaml \
     --concepts configs/concepts/shared.yaml configs/conditions/cardiometabolic.yaml \
     --data-root data/raw/cgmacros --out data/processed/cgmacros
   ```

`data/raw/`, `data/processed/` and `reports/local/` are git-ignored. Participant-level outputs, including the ingestion report, stay local.

**Known source caveats (CGMacros):**
- **HbA1c:** the toolkit accepts HbA1c in either % (DCCT/NGSP) or mmol/mol (IFCC) and stores each value in the unit it was measured in, with no conversion (the relation is affine, so values in different units must not be pooled without an explicit, reviewed conversion). The CGMacros data dictionary labels HbA1c as mmol/mol, but the dataset paper (Das et al. 2025) reports it as % (mean 6.16 ± 0.91 %; groups defined by <5.7, 5.7–6.4 and >6.4 %), so the dictionary label appears to be a documentation error. CGMacros HbA1c is ingested as `%`, unchanged.
- **CGM interpolation:** the dataset authors document that Libre (15-minute) and Dexcom (5-minute) readings were linearly interpolated to 1-minute intervals before release (Das et al. 2025, *Scientific Data*, doi:10.1038/s41597-025-05851-7). PGHD-Ready ingests the values as released and does not identify which individual values are original or interpolated. Analyses should not treat 1-minute CGM rows as independent sensor measurements.

## Guiding Principles

- Open-source and research-oriented.
- Modular, reusable, and configuration-first.
- Transparent and reproducible data transformations.
- Preservation of source context and measurement semantics.
- Leverage established standards rather than reinventing them.
- Evaluate data suitability before drawing predictive conclusions.

## Data and Licensing

The CGMacros dataset is subject to its own CC BY-NC-SA 4.0 licence and attribution requirements. Raw and participant-level data are not distributed in this repository.

The toolkit's software licence will be determined separately.