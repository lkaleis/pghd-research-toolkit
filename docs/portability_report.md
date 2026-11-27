# Portability report: BIG IDEAs Lab glycemic and wearable dataset

This report records what happened when PGHD-Ready was run on a second, independently collected dataset: what worked through configuration alone, what the source data turned out to contain, and what the run revealed about the toolkit. It is the portability test planned in the [roadmap](../README.md#project-status).

All figures are aggregate. No participant-level data, participant identifiers or free-text food entries are reproduced here. Raw and processed data stay local and are git-ignored.

## 1. Summary

- **The toolkit ingested and profiled a differently structured dataset through a new source configuration alone, with no new source-specific Python code.**
- The configuration needed three corrections after the first run. Each one came from a mismatch between the published description and the files themselves, and each was surfaced by the toolkit's reports rather than found by hand.
- The run exposed one toolkit bug, a completeness metric that was wrong whenever streams sit in separate files. It was fixed, with tests, and CGMacros results are unchanged.
- The breakfast prediction workflow (M3/M4) does not transfer without new research decisions, mainly because the food log has no meal labels and records food items rather than meals.

## 2. The dataset

**BIG IDEAs Lab Glycemic Variability and Wearable Device Data**, v1.1.3, PhysioNet. Open access, Open Data Commons Attribution License v1.0.

Cite: Cho, Kim, Bent & Dunn (2026), PhysioNet, doi:10.13026/aw6y-fc44; and Bent et al. (2021), *npj Digital Medicine* 4, 89, doi:10.1038/s41746-021-00465-w.

| | CGMacros (first dataset) | BIG IDEAs (this test) |
|---|---|---|
| Participants | 45 | 16 (post-menopausal women, HbA1c 5.2–6.4 %) |
| CGM | Libre Pro and Dexcom G6, interpolated by the authors to 1-minute rows | Dexcom G6 only, raw Dexcom Clarity export |
| Wearable | Fitbit Sense, one wide file with CGM | Empatica E4, one file per signal |
| File layout | One wide time-series file per participant | Separate files per participant and signal |
| Meals | Labelled meals (breakfast, lunch, …) with macronutrients | Food-item log, no meal labels |
| Participant data | Age, sex, BMI, HbA1c, lipids and more | Gender and HbA1c only |
| Dates | Shifted | Shifted |

**Scope of this test:** Dexcom glucose, E4 heart rate, the food log and demographics (about 240 MB). The high-frequency E4 signals (accelerometer, blood volume pulse, electrodermal activity, inter-beat interval, skin temperature; about 33 GB) were not downloaded, because the toolkit has no concepts for them yet.

## 3. What was needed

| Item | New code? | What was done |
|---|---|---|
| Source configuration | No | `configs/sources/big_ideas.yaml`, four kinds of table: Dexcom, E4 heart rate, food log, demographics |
| Concepts | No | Existing concepts reused: `glucose_interstitial`, `heart_rate`, `sex`, `hba1c` |
| Row filtering | No | The existing `where` filter keeps only glucose readings (`Event Type = EGV`) from the Clarity export |
| New event type | No | Food-log rows ingested as `food_item`, not `meal` (see §5) |
| Assumptions | No | Two recorded in the configuration: E4 heart rate is beats per minute, HbA1c is % (neither unit is stated by the source) |
| Toolkit fix | Yes | Completeness metric in profiling (see §6). A bug fix, not source-specific code |

## 4. Source issues found

The toolkit's reports surfaced each of these. None were silently corrected.

1. **One food-log file has no header row and three fewer columns** than the other fifteen, so its columns can't be identified. *Decision: excluded.* That participant's glucose and heart-rate data are still ingested.
2. **Dexcom timestamps don't match their own column header.** The header reads `YYYY-MM-DDThh:mm:ss`, but the values use a space instead of `T`. The first run therefore parsed no glucose at all; the ingestion report showed every glucose row as an unparseable timestamp. The format was corrected and all 16 files then parsed.
3. **One heart-rate file uses a different timestamp format** (`M/DD/YY h:mm`). It has no seconds, although the signal is recorded every second, and its month/day order can't be confirmed. *Decision: not reinterpreted.* Its rows are rejected with a reason and kept locally, so 15 of 16 participants have heart-rate data.
4. **Dexcom files carry metadata rows** (device and patient information, alerts) before the readings. These have no timestamp and are rejected as expected. The `Patient Info` column is dropped at read time and never ingested.
5. **Documentation gaps:** the source states no units for the E4 signals and no unit for HbA1c. Both are recorded as explicit assumptions in the configuration.

## 5. Results

### Ingestion

| | Count |
|---|---|
| Source records read | 9.4 million |
| Glucose readings (Dexcom G6) | 36,898 from 16 participants |
| Heart-rate readings (E4) | 8.75 million from 15 participants |
| Food items | 1,363 from 15 participants (median 88 per participant) |
| Participant attributes | gender and HbA1c for all 16 |
| Rejected records | 634,369: 634,188 from the heart-rate file in §4.3, 180 Clarity metadata rows, 1 food item with no start time |

**Nutrients:** energy, carbohydrate, protein and sugar are complete. Fat is empty for 109 food items and fibre for 111.

### Data quality (descriptive)

| | Dexcom G6 glucose | E4 heart rate |
|---|---|---|
| Expected interval (documented) | 300 s | 1 s |
| Observed span, median | 8.0 days | 8.1 days |
| Source rows with a value | 99.5 % | 100 % |
| Slot coverage, median (range) | **97 %** (74–100 %) | 86 % (20–96 %) |
| Gaps (participants affected) | 27 (13) | 133 (15) |
| Duplicate-timestamp rows | 0 | 88,419 |

How to read this:
- **Slot coverage** is the fraction of expected-interval slots across the observed span that contain at least one reading. It is not wear time.
- **Sampling origin** is `unknown` for every reading. The glucose data look like native 5-minute readings, unlike CGMacros, but the source doesn't state it and the toolkit never infers origin from a regular pattern.
- **Heart-rate duplicates** (88k rows sharing a timestamp) need explaining before heart rate is used analytically.

### Why the breakfast workflow does not transfer yet

The M3 breakfast task and the M4 experiment were designed on CGMacros. Running them here needs research decisions that have not been made:

- **No meal labels.** "Breakfast" would have to be defined, for example by clock time.
- **Food items, not meals.** About half of all food-log start times (49 %) have more than one item. Grouping items into meals is an analytical choice.
- **Missing predictors.** The approved M4 predictors include age and BMI, which this dataset doesn't contain.
- **One CGM device.** The task configuration expects both Libre and Dexcom.

## 6. Toolkit issue found and fixed

**Symptom:** the first quality profile reported that only 0.4 % of Dexcom source rows contained a value.

**Cause:** the completeness denominator was summed per participant across every source file, rather than per participant and stream. With glucose and heart rate in separate files, each participant's roughly 2,300 glucose readings were divided by the glucose file's rows plus about 600,000 heart-rate rows. CGMacros never exposed this, because both its streams come from one wide file.

**Fix:** the denominator now counts only the files each stream's own observations came from (`pghd_ready.core.profiling`).
- Two tests were added: separate files per stream, and one shared wide file.
- The full suite passes, including the real-data CGMacros tests.
- A fresh CGMacros profile is identical, row for row, to the saved one.

## 7. What this shows about the toolkit

**Worked as designed**
- A new source with a different layout, device and export format was added by configuration.
- Every defect surfaced in the ingestion or profiling reports, with counts and reasons. Rejected rows were preserved, not dropped.
- Undocumented units and ambiguous formats were recorded as assumptions or rejections, not guessed.

**Limitations exposed**
- **No file-exclusion option.** Excluding one file meant splitting a glob into two tables.
- **No header-less read option**, and only one timestamp format per table.
- **No item-to-meal aggregation.** Task definitions assume labelled meals.
- **Concept coverage.** There are no concepts yet for accelerometry, blood volume pulse, electrodermal activity, inter-beat interval or skin temperature.

## 8. Reproduce

Download only the files used here (about 240 MB) with the PhysioNet Python package, then run the same commands as for CGMacros:

```bash
physionet download big-ideas-glycemic-wearable --version 1.1.3 --output data/raw \
  --include "*/Dexcom_*.csv" --include "*/Food_Log_*.csv" --include "*/HR_*.csv" \
  --include "Demographics.csv" --include "LICENSE.txt"
mv data/raw/big-ideas-glycemic-wearable-1.1.3 data/raw/big_ideas

pghd-ready ingest --source configs/sources/big_ideas.yaml \
  --concepts configs/concepts/shared.yaml configs/conditions/cardiometabolic.yaml \
  --data-root data/raw/big_ideas --out data/processed/big_ideas
pghd-ready profile --m1 data/processed/big_ideas --out data/processed/big_ideas/quality
```

The aggregate outputs are `ingest_summary.json` and `quality/profile_report.md`. All other outputs are participant-level and must stay local.
