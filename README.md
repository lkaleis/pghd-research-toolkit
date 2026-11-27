# PGHD-Ready

**An open-source research toolkit for integrating, assessing, and establishing the predictive fitness of patient-generated health data (PGHD).**

## Overview

> **New here?** Start with the [end-to-end demonstration](demo/README.md), a five-minute tour from raw PGHD to a participant-validated prediction experiment on CGMacros, with [architecture diagrams](demo/architecture.md).

Patient-generated health data (PGHD) come from wearables, connected sensors, health apps, self-tracking tools and patient-reported outcomes. They offer real opportunities for spotting emerging health risks and strengthening real-world evidence.

However, PGHD are often fragmented across sources, inconsistent in structure, variable in quality, and hard to align in time. These problems can undermine downstream statistical analyses and predictive models.

**PGHD-Ready** provides reusable methods for harmonising heterogeneous PGHD, evaluating data quality, and assessing whether the data are fit for a specific predictive task.

Instead of building a new pipeline for every dataset or condition, the toolkit is **configuration-first and modular**. Data sources and research applications are described in configuration files, while provenance and clinical meaning are preserved.

## Research Focus

The central question is how the characteristics and quality of PGHD affect their suitability for finding predictive clinical signals. The main areas are:

- **Data integration:** harmonising sources, schemas, timestamps, units and clinical concepts.
- **Data quality:** completeness, missingness, temporal coverage, irregular sampling, and consistency across modalities.
- **Temporal alignment:** how differences in measurement frequency and event timing affect reliability.
- **Predictive fitness:** how data quality, modality availability and integration choices affect suitability for a given prediction task.
- **Reproducibility:** transparent mappings, transformation records, validation rules, and machine-readable quality assessments.

## Architecture

The toolkit has three parts:

1. **Shared core:** ingestion, schema mapping, standardisation, validation, provenance tracking, temporal alignment and data-quality assessment, used the same way across datasets.
2. **Condition-specific modules:** clinical concepts and analysis methods for a particular health area.
3. **Data source integrations:** configuration-based mappings for common formats, with optional custom adapters for sources that need special handling.

New sources should be added through configuration rather than new Python code where possible. Existing standards and open-source tools are reused where appropriate. The goal is that researchers can reuse the same core across datasets and conditions rather than rebuilding data preparation for each one.

## PGHD Domains

PGHD-Ready is a cross-condition framework, not a single-disease tool. It describes data using four broad domains:

| Domain | Examples |
|---|---|
| **Physiological** | Continuous and intermittent glucose, heart rate, blood pressure, weight and body composition, device biomarkers |
| **Behavioural** | Meals and nutrition, physical activity, sleep |
| **Medication-related** | Prescribed regimens, patient-reported medication-taking, missed or delayed doses, reasons for non-adherence |
| **Patient-reported** | Symptoms, side effects, patient-reported outcomes, questionnaires, free-text journals |

Domains describe the data. They are not separate software modules, and one stream can belong to several domains. The shared core treats all domains alike. Clinical concepts, derived measures and prediction tasks live in the condition-specific modules. The project does not define a universal healthcare data model. New concepts are added through configuration when a research question needs them.

## Initial Research Application

The first application is cardiometabolic health, using the **CGMacros** dataset. It asks how data quality, measurement characteristics, temporal alignment and modality availability affect the reliability and predictive suitability of integrated PGHD. Portability was then tested on a second, independently collected dataset (BIG IDEAs Lab, Dexcom G6 and Empatica E4): ingestion and quality profiling ran through a new configuration file, with no source-specific code. See the [portability report](docs/portability_report.md).

## Longer-Term Direction

The aim is to extend the toolkit to other health domains, such as behavioural health (mood, sleep, symptoms, medication adherence). A planned research question is whether patient-reported medication adherence adds predictive value for cardiometabolic outcomes beyond physiological and behavioural PGHD. CGMacros has no medication records, so this needs a separate real-world dataset.

## Intended Outputs

- Standardised, analysis-ready datasets
- Reusable source mappings and configuration templates
- Machine-readable data-quality profiles
- Task-specific predictive-fitness assessments
- Provenance and transformation records
- Reproducible analytical reports

## Project Status

**Early research and development.** This is an exploratory research project, not a clinically validated decision-support system.

| Capability | Status |
|---|---|
| Common data representation (observations, events, participant attributes, streams) with schema validation | Implemented |
| Annotation of documented issues, suspected anomalies, ambiguities and assumptions | Implemented |
| Sampling origin per observation (native / interpolated / unknown; default unknown) with explicit inclusion policies | Implemented generically. Row-level classification of CGM values is not planned |
| Shared and condition-specific concept registries with optional PGHD domain tags | Implemented. The four domain terms are fixed in code |
| Configuration-driven CSV ingestion with provenance, rejected-record preservation and a plain-language ingestion report | Implemented, tested on a small synthetic example |
| Portability to a second real dataset (BIG IDEAs Lab: Dexcom G6, Empatica E4 heart rate, food log) | Ingestion and profiling run by configuration only, locally. See the [portability report](docs/portability_report.md) |
| CGMacros ingestion (CGM, Fitbit, meals, participant attributes), checked against an independent reference profiler | Implemented, run locally only |
| Definition-based consistency checks (first rule: non-HDL = total cholesterol − HDL) with reviewed approval | Implemented. Run on CGMacros in review mode (aggregate only) |
| Descriptive data-quality profiling: completeness, timestamp coverage, sampling intervals, gaps, sampling origin, per participant, stream and day | Implemented, run locally on CGMacros |
| Plausibility rules and cross-device (Libre vs Dexcom) agreement | Planned |
| Analytical preparation: time windows, event alignment, attribute joins, eligibility ledger. First task: breakfast glucose response (signed iAUC per CGM device) | Implemented for one task, run locally on CGMacros. No modelling at this stage |
| Predictive demonstration: breakfast iAUC from meal composition, baseline glucose and participant characteristics, using naive, linear, random-forest and ridge models with leave-one-participant-out validation per device | Complete, internal validation only. See the [modelling report](docs/modelling_report.md) |
| JSON, FHIR or open mHealth inputs; free-text sources | Not implemented |
| Medication-related and patient-reported data representations | Not implemented (see roadmap) |

**Roadmap** (each step is reviewed before the next begins):

1. **M1** Foundation, ingestion engine and CGMacros ingestion. *(Complete.)*
2. **M2** Standalone data-quality assessment. *(Consistency checks and descriptive profiling complete. Plausibility rules and cross-device agreement planned.)*
3. **M3** Temporal alignment and task fitness. *(Breakfast glucose-response preparation complete. No further tasks planned yet.)*
4. **M4** Bounded predictive demonstration. *(Complete. Preferred model: linear regression without total energy. See the [modelling report](docs/modelling_report.md).)* A second synthetic source is now treated as supporting test infrastructure, not a separate milestone.
5. **M5** Medication adherence and multimodal PGHD: a medication event representation that keeps prescribed regimens separate from patient-reported taking events, configurable ingestion of medication-tracking exports, and a synthetic adherence example.

## Trying the ingestion engine

A small, fictional dataset is included in `data/synthetic/wearable_demo/`. Its README documents each intentional data problem.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pghd-ready ingest \
  --source    data/synthetic/wearable_demo/config/source.yaml \
  --concepts  configs/concepts/shared.yaml configs/conditions/cardiometabolic.yaml \
  --data-root data/synthetic/wearable_demo/raw \
  --out       data/processed/wearable_demo
```

The run writes standardised Parquet tables, audit files and `ingestion_report.md`, which explains in plain language what was read, mapped, transformed, flagged or rejected. A successful run does not mean the data are fit for any particular analysis.

## Profiling data quality (M2b)

`pghd-ready profile` describes a finished ingestion run. It never modifies, interpolates, resamples or imputes data, and it sets no thresholds.

```bash
.venv/bin/pghd-ready profile --m1 data/processed/cgmacros --out data/processed/cgmacros/quality
# optional, for a stream with no documented interval:  --expected-interval fitbit_hr=60   (and --gap-factor, default 1.5)
```

**Outputs:** an aggregate `profile_report.md`, a `profile_manifest.json` (input and output hashes, code commit, parameters), and participant-level tables that must stay local. Metric definitions are in `src/pghd_ready/core/profiling.py`.

**Key limitations:**
- **Observed span is not wear time.** It is the interval between the first and last timestamp. It does not show the device was worn.
- **Gaps are assessed only where the expected interval is documented** (in the source config or via `--expected-interval`). It is never inferred from the data. For CGMacros only Libre (900 s) and Dexcom (300 s) qualify, so Fitbit gaps are reported as "not assessed".
- **CGMacros CGM values were interpolated by the dataset authors to 1-minute rows.** Use slot coverage at the documented interval, not counts of 1-minute rows.
- **Dates are shifted** in CGMacros, so calendar days are not real dates.

## Preparing an analysis table (M3)

`pghd-ready prepare` runs a task definition on a finished ingestion run. It writes a per-candidate eligibility ledger, window readings and an analysis table (all participant-level, kept local), plus an aggregate `task_report.md` and manifest. No model is fitted and nothing is resampled or imputed.

```bash
.venv/bin/pghd-ready prepare --task configs/tasks/postprandial_breakfast.yaml \
  --m1 data/processed/cgmacros --out data/processed/cgmacros/tasks/postprandial_breakfast
```

Generic mechanics are in `core/alignment.py` and `core/eligibility.py`. The task-specific definitions (outcome, baseline, meal separation, predictors, calculability rules) are in the task config and `conditions/cardiometabolic/tasks/postprandial.py`.

**Key limitations of the breakfast workflow:**
- **Thresholds are operational, not clinically validated.** They derive from the documented device sampling intervals (for example, a minimum of 1 baseline reading for Libre and 3 for Dexcom).
- **The data were interpolated before release,** so the thresholds cannot detect gaps filled in beforehand, and the iAUC may partly reflect that interpolation.
- **iAUC is signed** (area below baseline is not clamped) and reported per device. Devices are not compared.
- **The outcome is dataset-derived, not a measured physiological response.** Describe it as "calculable iAUC from CGMacros readings".
- **The baseline may contain post-meal information.** Linear interpolation towards the first reading after the meal can leak into the last 15 min (Libre) or 5 min (Dexcom) of the baseline window. Any use of the baseline as a predictor must account for this.
- **Non-calculable breakfasts are not missing at random.** They are excluded for device-specific reasons, so the calculable sets differ by device.

## Breakfast iAUC prediction experiment (M4)

A small, single-purpose experiment on the M3 analysis table, not a general ML framework. It predicts breakfast iAUC from meal composition, baseline glucose, HbA1c, BMI and age, using leave-one-participant-out validation per device. The random forest needs an optional extra (`pip install -e ".[experiment]"`).

**Summary of findings:** on CGMacros (45 participants; 422 Libre and 392 Dexcom calculable breakfasts, analysed separately), linear regression reduced participant-macro MAE by about a quarter to a third compared with a naive mean. The preferred specification is linear regression **without total energy** (Model B). It is no more accurate than the original model but removes severe collinearity. Fasting glucose and ridge regression added no detectable value. These are internal-validation results on a dataset-derived outcome, with **no clinical claims**.

Methods, commands, diagnostics, full results and limitations are in the **[modelling report](docs/modelling_report.md)**.

## Development and Tests

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt -e ".[dev]"
env -u PYTHONPATH .venv/bin/python -m pytest
```

- Synthetic tests always run, using only the fictional dataset and data generated inside the tests.
- **Real-data reconciliation tests** (`tests/test_cgmacros_real.py`, marker `real_data`) run only where CGMacros is available locally, and are otherwise skipped. They compare the ingestion engine with an independent reference profiler, with aggregate-only assertions, and add about 40 seconds. Run only them with `pytest -m real_data`, or exclude them with `pytest -m "not real_data"`.

**The CGMacros dataset is not included and must never be committed.** To use it locally:

1. Download CGMacros v1.0.0 from [PhysioNet](https://doi.org/10.13026/3z8q-x658) (CC BY-NC-SA 4.0) and extract it to `data/raw/cgmacros/` (one `CGMacros-0XX/` folder per participant, plus `bio.csv`).
2. Download the `DataDictionary_*.csv` files from the same page into `data/raw/cgmacros_docs/`.
3. Run the ingestion:
   ```bash
   .venv/bin/pghd-ready ingest --source configs/sources/cgmacros.yaml \
     --concepts configs/concepts/shared.yaml configs/conditions/cardiometabolic.yaml \
     --data-root data/raw/cgmacros --out data/processed/cgmacros
   ```

`data/raw/`, `data/processed/` and `reports/local/` are git-ignored. Participant-level outputs stay local.

**Known source caveats (CGMacros):**
- **HbA1c units:** the toolkit stores HbA1c in the unit it was measured in (% or mmol/mol) with no conversion, because values in different units must not be pooled without an explicit, reviewed conversion. The CGMacros data dictionary labels HbA1c as mmol/mol, but the dataset paper (Das et al. 2025) reports %, so the label appears to be a documentation error. It is ingested as `%`, unchanged.
- **CGM interpolation:** the authors linearly interpolated Libre (15-min) and Dexcom (5-min) readings to 1-minute intervals before release (Das et al. 2025, *Scientific Data*, doi:10.1038/s41597-025-05851-7). The toolkit ingests values as released and cannot tell original from interpolated values, so 1-minute rows should not be treated as independent sensor measurements.

## Guiding Principles

- Open source and research-oriented
- Modular, reusable and configuration-first
- Transparent, reproducible transformations
- Preservation of source context and measurement semantics
- Established standards over reinvention
- Evaluate data suitability before drawing predictive conclusions

## Data and Licensing

CGMacros is subject to its own CC BY-NC-SA 4.0 licence and attribution requirements. Raw and participant-level data are not distributed here. The toolkit's software licence will be determined separately.
