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

## Initial Research Application

The first implementation focuses on cardiometabolic health, using the CGMacros dataset to investigate how data quality, measurement characteristics, temporal alignment, and modality availability influence the reliability and predictive suitability of integrated PGHD.

A differently structured synthetic digital health dataset will subsequently be used to evaluate the portability and reusability of the ingestion architecture.

## Longer-Term Direction

The longer-term ambition is to extend PGHD-Ready across multiple health domains, potentially including behavioral health, while maintaining a shared foundation for integrating and assessing heterogeneous patient-generated data.

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

**Early research and development.**

The initial dataset has been profiled and the architecture is being designed. Core functionality, validation, and research experiments will be developed incrementally.

This is an exploratory research initiative, not a clinically validated decision-support system.

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