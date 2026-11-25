# PGHD-Ready: an end-to-end demonstration

A five-minute tour of what the PGHD Research Toolkit does, followed by a worked example on a public dataset (CGMacros). It runs from raw files to a participant-validated prediction experiment, and every number below is an aggregate result produced by the code in this repository.

- **Architecture diagrams:** [architecture.md](architecture.md)
- **Project overview and development notes:** [the main README](../README.md)

## 1. The problem

Patient-generated health data (PGHD) come from continuous glucose monitors, fitness trackers, meal logs, apps and questionnaires. Before anyone can ask a research question of such data, four hard problems have to be solved:

| Problem | Example from CGMacros |
|---|---|
| **Heterogeneity** | Two glucose sensors, a fitness tracker, meal logs and laboratory results in different files, columns and units |
| **Undocumented quirks** | Dates shifted for privacy. Glucose readings interpolated to one-minute rows by the dataset authors. A data-dictionary unit label for HbA1c that contradicts the dataset paper |
| **Quality is not fitness** | A file can be read perfectly and still be unsuitable for a given analysis, for example when there are too few readings around a meal |
| **Reproducibility** | Every exclusion, transformation and modelling choice must be traceable, or results cannot be checked |

PGHD-Ready makes each of these steps explicit, configurable, reviewable and reproducible, instead of burying them in one-off scripts.

## 2. The toolkit in five stages

| Stage | Question it answers | Toolkit component | Main outputs |
|---|---|---|---|
| **M1 Integration** | Can the source be read into one common, traceable representation? | `pghd-ready ingest`, [`core/ingest.py`](../src/pghd_ready/core/ingest.py) | Observations, events, participant attributes, streams; rejected records; manifest; ingestion report |
| **M2a Consistency** | Do related values agree with a published definition? | [`core/consistency.py`](../src/pghd_ready/core/consistency.py) (Python API) | Coverage counts and per-record findings (in memory) |
| **M2b Profiling** | How complete, regular and well covered is each data stream? | `pghd-ready profile`, [`core/profiling.py`](../src/pghd_ready/core/profiling.py) | Stream, day and gap tables; aggregate profile report |
| **M3 Preparation** | Which events are eligible for a specific research task, and what is the outcome? | `pghd-ready prepare`, [`core/alignment.py`](../src/pghd_ready/core/alignment.py), [`core/eligibility.py`](../src/pghd_ready/core/eligibility.py) | Eligibility ledger and funnel; analysis table; task report |
| **M4 Prediction** | How well can the outcome be predicted for patients the model has not seen? | `python -m` experiment modules | Metrics with uncertainty, model metadata, aggregate report, manifest |

Each stage reads the previous stage's outputs without modifying them, and writes a manifest of file hashes ([lineage diagram](architecture.md#provenance-and-lineage)).

## 3. Three layers: what is reusable and what is research-specific

| Layer | What it contains | Where |
|---|---|---|
| **Reusable toolkit** | Ingestion engine, common representation, provenance, sampling origin, consistency engine, profiling, event windows, eligibility ledger | [`src/pghd_ready/core/`](../src/pghd_ready/core/), [`configs/concepts/shared.yaml`](../configs/concepts/shared.yaml) |
| **Cardiometabolic** | Glucose, lipid and anthropometric concepts and units; meal event schema; non-HDL derivation rule and its verified reference | [`configs/conditions/cardiometabolic.yaml`](../configs/conditions/cardiometabolic.yaml), [`configs/quality/`](../configs/quality/), [`src/pghd_ready/conditions/cardiometabolic/`](../src/pghd_ready/conditions/cardiometabolic/) |
| **CGMacros study** | Source mapping, the dataset's consistency-rule application, the breakfast task and the two experiments | [`configs/sources/cgmacros.yaml`](../configs/sources/cgmacros.yaml), [`configs/tasks/`](../configs/tasks/), [`configs/experiments/`](../configs/experiments/), [`…/cardiometabolic/tasks/`](../src/pghd_ready/conditions/cardiometabolic/tasks/), [`…/cardiometabolic/experiments/`](../src/pghd_ready/conditions/cardiometabolic/experiments/) |

The core layer never refers to a dataset, device or condition, and an automated test enforces this. CGMacros is integrated through configuration alone, with no source-specific Python code.

## 4. Worked example: CGMacros

**The dataset.**
- **Contents:** [CGMacros v1.0.0](https://doi.org/10.13026/3z8q-x658) (PhysioNet; Das et al. 2025, *Scientific Data*, [doi:10.1038/s41597-025-05851-7](https://doi.org/10.1038/s41597-025-05851-7)) follows 45 adults for about ten days each. It records:
  - two continuous glucose monitors worn at the same time (Abbott FreeStyle Libre Pro, Dexcom G6 Pro);
  - a Fitbit;
  - logged meals with macronutrients, including designed breakfasts;
  - laboratory and anthropometric measurements.
- **Access:** the dataset is open access under CC BY-NC-SA 4.0. It is **not** redistributed here.

### M1: integration and harmonisation

The source configuration maps each column to a concept, a unit and, where documented, a device limit or error code.

| Input | Output |
|---|---|
| 46 files, 687,580 data rows, 0 rejected | **3,243,018 observations** in 7 streams (Libre 687,360; Dexcom 629,825; Fitbit heart rate 610,256; and four Fitbit activity streams) |
| | **4,903 events:** 1,706 meals (436 breakfasts, 435 lunches, 492 dinners, 342 snacks, 1 without a meal type) and 3,197 meal-photo events |
| | **855 participant attributes** across 17 concepts, each stored in the unit it was measured in |

What M1 preserves and records, rather than silently "cleaning":
- **Raw values and source rows.** Every standardised record keeps its raw value, source file and source row.
- **Flags, not deletions.** For example:
  - 13,542 glucose readings at a documented reporting limit (Libre 13,027, Dexcom 515);
  - 9 Libre readings outside documented device limits;
  - two Fitbit streams whose semantics are undocumented;
  - 2 documented laboratory error codes.
- **Sampling origin.** Each observation is recorded as native, interpolated or unknown. All CGMacros readings are recorded as `unknown`, because the authors document interpolation but the rows do not say which values are original.
- **Provenance.** A manifest holds the SHA-256 hash of every input and output file, plus an analyst-readable ingestion report.

### M2a: definition-based consistency (non-HDL cholesterol)

Non-HDL cholesterol is defined as total cholesterol minus HDL cholesterol (NCEP ATP III, 2002). The toolkit checks the reported non-HDL value against that definition, allowing for the reporting resolution of each value. Before any check runs, it requires:
1. a fully verified reference;
2. technical approval bound to a content hash of the rule;
3. a run authorisation for this specific dataset, bound to a hash of its application and assumptions.

| Candidates | Evaluated | Consistent | Discrepancies |
|---|---|---|---|
| 45 | 45 | **45** | 0 |

This is computational conformance (the stored values follow the definition), not a clinical validation of the measurements.

### M2b: descriptive quality profiling

Profiling describes what was ingested. It sets no thresholds and changes no data. Medians are taken across participants.

| Stream | Participants | Observations | Observed span, days | Slot coverage at documented interval | Gaps longer than 1.5× interval |
|---|---|---|---|---|---|
| Libre Pro (15-min device interval) | 45 | 687,360 | 10.6 | 1.000 | 30, in 2 participants |
| Dexcom G6 Pro (5-min device interval) | 45 | 629,825 | 9.9 | 1.000 | 277, in 9 participants |
| Fitbit heart rate (no documented interval) | 45 | 610,256 | 10.2 | not assessed | not assessed |

How to read this table:
- **Observed span is not wear-time.** It is the time between the first and last reading.
- **Gaps are assessed only against a documented interval.** Intervals are never guessed from the data, so the Fitbit stream is reported as "not assessed".
- **Slot coverage uses the device interval, not one-minute rows.** It counts how many of the device's 15- or 5-minute slots contain a reading, so the authors' interpolation to one-minute rows cannot inflate it.

### M3: from events to an analysis-ready dataset

The research task is the glucose response to breakfast, measured as the **signed incremental area under the curve (iAUC)** over the 120 minutes after the meal. This is the area between the glucose curve and the pre-meal baseline (the mean over the previous 15 minutes); area below the baseline counts as negative. Each device is handled separately.

```text
436 logged breakfasts
 └─ 2 excluded: another meal in the 60 min before
 └─ 3 excluded: another meal in the 120 min after
431 eligible breakfasts (per device)
 ├─ Libre:  422 with a calculable iAUC   (9 not: 8 no reading at +120 min, 1 none at the meal time)
 └─ Dexcom: 392 with a calculable iAUC   (39 not: 19 too few baseline readings, 17 no reading at +120 min,
                                          2 no readings in the window, 1 interval longer than 5 min)
```

| | Libre (422) | Dexcom (392) |
|---|---|---|
| Signed iAUC, median [IQR], mg/dL × min | 2,903 [1,249 – 5,634] | 3,065 [1,215 – 6,259] |
| Breakfasts with negative iAUC | 29 | 40 |

Every candidate breakfast appears in an eligibility ledger with the first rule that excluded it, so the funnel above can be reproduced exactly. The calculability rules are operational choices based on the documented device intervals; they are not clinically validated thresholds.

### M4: participant-aware prediction

**Question.** Can meal composition, pre-meal baseline glucose and participant characteristics (HbA1c, BMI, age) predict the breakfast iAUC for a person the model has never seen?

**How it was tested.**
- **Validation:** leave-one-participant-out. Each of the 45 participants is held out in turn, with *all* their breakfasts, and the model is trained on the other 44. The two devices are analysed independently and never pooled.
- **Primary metric:** participant-macro MAE. This is the average size of the prediction error for each held-out participant, averaged across participants so that people with more breakfasts do not dominate.
- **Uncertainty:** 95% intervals from resampling whole participants. Comparisons between models are paired on the same resamples.
- **Baseline for comparison:** a "naive" model that always predicts the average training iAUC.

**Original experiment.** Participant-macro MAE in mg/dL × min (lower is better), with 95% intervals:

| Model | Libre | Dexcom |
|---|---|---|
| Naive (training mean) | 2,987 [2,551, 3,491] | 3,303 [2,794, 3,912] |
| Linear regression, 9 predictors | 2,088 [1,730, 2,541] | 2,455 [2,080, 2,893] |
| Random forest, fixed settings | 2,331 [1,925, 2,815] | 2,849 [2,413, 3,337] |

The linear model reduces the average error by about a quarter to a third compared with the naive model: 899 [507, 1,317] on Libre and 848 [450, 1,286] on Dexcom. It explains about 40% of the squared error that the naive model leaves (R² relative to naive 0.42 and 0.40). The random forest was not better than linear regression. Individual predictions remain imprecise: the typical error is a sizeable fraction of a typical response.

**Refinement.** The original model includes total energy, which is almost a sum of the macronutrients, so the predictors are strongly collinear. A second experiment, on the same breakfasts and folds, compared:

| Model | Libre | Dexcom |
|---|---|---|
| A: original linear model (reference) | 2,088 | 2,455 |
| **B: A without total energy (preferred)** | **2,093** | **2,451** |
| C: B + study-start fasting glucose | 2,131 | 2,508 |
| D1 / D2: ridge regression on B / C | 2,088 / 2,091 | 2,427 / 2,458 |

None of these differences is distinguishable from zero, so none of the refinements predicted better or worse.

**Final model decision: Model B.** B is not more accurate than A (paired difference −5 [−22, 13] on Libre and 4 [−1, 9] on Dexcom). It is preferred because it is the more defensible specification:
- **Less collinear:** the largest variance inflation factor falls from 59 (Libre) and 156 (Dexcom) to about 2.
- **More robust coefficients:** in A, the energy coefficient has opposite signs on the two devices. In B, every coefficient has the same sign on both devices, and its macronutrient coefficients vary much less across folds.
- **No added value from fasting glucose or ridge:** neither showed a detectable improvement. Fasting glucose is also strongly correlated with HbA1c (Spearman 0.87).

**Prediction is not explanation.** A model can predict well while its individual coefficients are unstable, because correlated predictors can trade weight with little change in the predictions. None of the coefficients is an independent, causal or clinical effect.

## 5. What should and should not be generalised

| Reusable beyond this example | Specific to this example, not to be generalised |
|---|---|
| Configuration-first ingestion with preserved raw values and provenance | The CGMacros mappings, flags and documented-issue handling |
| Approval-gated consistency rules bound to verified references | The non-HDL result (45/45) for this dataset |
| Descriptive profiling against documented intervals only | The coverage and gap figures above |
| Event windows, eligibility ledgers and funnels | The breakfast definitions and calculability thresholds |
| Participant-aware validation, participant bootstrap and hash-linked lineage | Every model result, the choice of Model B and any statement about predictors |

## 6. Limitations

- **Small cohort:** 45 participants, so the intervals are wide. Participant characteristics rest on 45 distinct values.
- **Internal validation only:** leave-one-participant-out within CGMacros is not external validation.
- **Designed breakfasts:** standardised breakfasts, not unrestricted everyday meals.
- **Interpolated glucose data:** the released CGM values were interpolated to one-minute rows by the dataset authors, which may affect both the pre-meal baseline and the iAUC.
- **Dataset-derived outcome:** the signed iAUC is an analytical construct, not a clinically validated measure of glucose response.
- **No demonstrated clinical utility** or generalisability to other populations, devices, conditions or meal settings.
- **Toolkit scope:** delimited-text inputs only, one consistency rule, no plausibility rules or Libre–Dexcom agreement yet, and one task on one real dataset ([full list](architecture.md#what-is-and-is-not-implemented)).

## 7. Reproduce the example

**Install** (Python 3.11 or later):

```bash
git clone <this repository> && cd pghd-research-toolkit
python -m venv .venv && .venv/bin/pip install -r requirements.txt -e ".[dev,experiment]"
env -u PYTHONPATH .venv/bin/python -m pytest          # synthetic tests always run; CGMacros tests skip without the data
```

The `experiment` extra (scikit-learn) is needed only for the random forest.

**Try it without any download.** The repository ships a small, entirely fictional dataset with deliberate problems:

```bash
.venv/bin/pghd-ready ingest --source data/synthetic/wearable_demo/config/source.yaml \
  --concepts configs/concepts/shared.yaml configs/conditions/cardiometabolic.yaml \
  --data-root data/synthetic/wearable_demo/raw --out data/processed/wearable_demo
```

**Get CGMacros.**
1. Download CGMacros v1.0.0 from [PhysioNet](https://doi.org/10.13026/3z8q-x658). It is open access under the CC BY-NC-SA 4.0 licence; follow the citation instructions on the PhysioNet page.
2. Extract it to `data/raw/cgmacros/`, with one `CGMacros-0XX/` folder per participant plus `bio.csv`.
3. Put the official `DataDictionary_*.csv` files in `data/raw/cgmacros_docs/`.

**Run the stages:**

```bash
# M1  integration
.venv/bin/pghd-ready ingest --source configs/sources/cgmacros.yaml \
  --concepts configs/concepts/shared.yaml configs/conditions/cardiometabolic.yaml \
  --data-root data/raw/cgmacros --out data/processed/cgmacros

# M2b profiling
.venv/bin/pghd-ready profile --m1 data/processed/cgmacros --out data/processed/cgmacros/quality

# M3  breakfast analysis table
.venv/bin/pghd-ready prepare --task configs/tasks/postprandial_breakfast.yaml \
  --m1 data/processed/cgmacros --out data/processed/cgmacros/tasks/postprandial_breakfast

# M4  original experiment, then the refinement
.venv/bin/python -m pghd_ready.conditions.cardiometabolic.experiments.breakfast_iauc \
  --config configs/experiments/breakfast_iauc.yaml \
  --task-dir data/processed/cgmacros/tasks/postprandial_breakfast \
  --out data/processed/cgmacros/experiments/breakfast_iauc
.venv/bin/python -m pghd_ready.conditions.cardiometabolic.experiments.breakfast_iauc_refinement \
  --config configs/experiments/breakfast_iauc_refinement.yaml \
  --task-dir data/processed/cgmacros/tasks/postprandial_breakfast --m1 data/processed/cgmacros \
  --reference-dir data/processed/cgmacros/experiments/breakfast_iauc \
  --out data/processed/cgmacros/experiments/breakfast_iauc_refinement
```

M2a has no command-line entry point yet. It runs in memory through the Python API, and `evaluate` refuses to run unless every approval gate passes:

```python
import pandas as pd
from pghd_ready.core import consistency as cs

refs = cs.load_references("configs/quality/references.yaml")
rule = cs.load_rules("configs/quality/rules/cardiometabolic.yaml")["CM-CONS-NONHDL-001"]
(app,) = cs.load_applications("configs/quality/applications/cgmacros.yaml")
attrs = pd.read_parquet("data/processed/cgmacros/participant_attributes.parquet")

result = cs.evaluate(rule, app, refs, attrs)
print(result.coverage[["n_candidates", "n_evaluated", "n_consistent", "n_discrepancies"]])
```

**What to expect.**
- **Aggregate outputs:** `ingestion_report.md`, `profile_report.md`, `task_report.md`, `experiment_report.md` and `refinement_report.md`, each with a manifest of input and output hashes and the code commit.
- **Determinism:** reruns of M1 and of both M4 experiments were checked to produce byte-identical data outputs.
- **Keep participant-level outputs local.** The standardised tables, ledgers, windows, analysis tables, fold assignments and predictions contain participant-level data. They are written under the git-ignored `data/processed/` and must not be shared or committed.
