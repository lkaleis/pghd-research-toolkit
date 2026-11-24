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
| CGMacros base ingestion (CGM, Fitbit, meals, participant attributes) with independent reconciliation tests | Implemented, run locally only (data not distributed); under review |
| Data-quality profiling (coverage, gaps, sampling, plausibility, cross-stream agreement) | Planned |
| Temporal alignment and task-specific predictive-fitness assessment | Planned |
| JSON, FHIR or open mHealth inputs; free-text sources | Not implemented |
| Medication-related and patient-reported data representations | Not implemented (see roadmap) |

Roadmap (each step is reviewed before the next begins):

1. **M1** Foundation, generic ingestion engine and CGMacros ingestion.
2. **M2** Standalone data-quality profiling.
3. **M3** Temporal alignment and task fitness.
4. **M4** A second, differently structured synthetic source to test reuse.
5. **M5** Initial experiment on how data limitations affect a predictive task.
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