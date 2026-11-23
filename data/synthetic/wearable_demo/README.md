# `wearable_demo`: a fictional, deliberately messy PGHD export

**Everything in this folder is invented.** No real person, device, study or
dataset is represented. Values were written by hand to exercise the ingestion
engine; they are not physiologically meaningful and must not be analysed.

The dataset is small enough to read in a text editor. It is the reference
example for configuration-driven ingestion (milestone M1b) and is also used by
the automated tests.

## Structure

```text
wearable_demo/
├── README.md                          this file (includes the fictional data dictionary)
├── config/
│   └── source.yaml                    source configuration for the generic engine (no Python code)
└── raw/
    ├── device_export/                 one wide CSV per participant from a fictional wristband + two patch sensors
    │   ├── user_01/user_01.csv
    │   ├── user_02/user_02.csv
    │   └── user_03/user_03.csv
    ├── app_export/app_glucose.csv     long-format export from a fictional phone app (one row per reading)
    └── profile.csv                    one row per participant: anthropometrics, a lab value, a fingerstick
```

Concepts and units come from the shared registry
`configs/conditions/cardiometabolic.yaml`; no separate concept file is needed.

## Fictional participants

| Local ID | Device export | App export | Profile |
|---|---|---|---|
| 1 | `user_01` | yes | yes |
| 2 | `user_02` | yes | yes |
| 3 | `user_03` | yes | yes |
| 4 | - | - | yes (profile only) |

Standardised IDs are `wearable_demo:001` ... `wearable_demo:004`.

## Fictional data dictionary

This section plays the role of the provider's documentation. The configuration
cites it as evidence for its documented transformations.

| File / column | Meaning |
|---|---|
| device export `Timestamp` | Local clock time, minute resolution, no timezone. *The timezone is not documented.* |
| `Sensor A`, `Sensor B` | Glucose from two patch sensors, mg/dL |
| `HR` / `Heart Rate` | Heart rate, beats per minute |
| `Activity x10` | MET multiplied by 10 |
| `Intensity` | *Not documented* |
| `Meal`, `Kcal`, `Carbs`, `Photo` | Meal label, energy (kcal), carbohydrate (g), photo path |
| `Amount` | Amount consumed, documented as 0-100 % |
| app export `recorded_at` | ISO-8601 time with UTC offset |
| app export `value` | Glucose in mmol/L when `type` = `glucose` |
| profile `Weight (lb)`, `Height` | Pounds, inches |
| profile `LDL` | mg/dL; the value **999 means "calculation error"** |
| profile second `Time` column | Clock time of the fingerstick; the date is not recorded |

## Intentional imperfections

| Where | Imperfection | What the engine is expected to do |
|---|---|---|
| `user_01.csv` | UTF-8 byte-order mark, stray `Unnamed: 0` index column | strip the BOM; drop the column as configured |
| `user_01.csv` | header `Sensor B ` with a trailing space | match the alias `Sensor B` |
| `user_01.csv` | quoted `Note` spanning two lines (lines 4-5) | read it as one record; later line numbers stay correct |
| `user_01.csv` | blank line 7 | skip it; count it |
| `user_01.csv` line 8 | empty timestamp | reject the row (`missing_timestamp`), keep its content |
| `user_01.csv` line 9 | one field too many | reject the row (`field_count_mismatch`), keep its fields |
| `user_01.csv` | `Amount` = 300 | keep the value; flag `amount_outside_documented_range` (ambiguous) |
| `user_01.csv` | `Note` column | not configured, so it is reported as an unused column |
| `user_02.csv` | columns in a different order; `Heart Rate` instead of `HR` | resolve columns by name or alias |
| `user_02.csv` | `Intensity` instead of `Activity x10`; no `Amount` column | missing optional fields are reported, not guessed |
| `user_02.csv` line 5 | `2000-03-26 02:30:00` does not exist in Europe/Berlin (DST gap) | reject the row (`timestamp_dst_unresolvable`) |
| `user_02.csv` line 6 | meal with `Kcal` = `about 700` | keep the event; energy empty, `unparseable_numeric`, original in `raw_attributes` |
| `user_03.csv` | sensor value `HIGH` | keep `raw_value`; value empty; `unparseable_numeric` |
| `user_03.csv` line 3 | timestamp `not a time` | reject the row (`unparseable_timestamp`) |
| `user_03.csv` | meal label `brunch` | keep `raw_label`; no category; `unmapped_category` |
| `user_03.csv` line 5 | row with fewer fields than the header | pad with empty values; count it |
| `app_glucose.csv` | glucose in mmol/L, times with `Z` / `+01:00` offsets | convert to mg/dL (concept-specific factor) and to UTC |
| `app_glucose.csv` | `type` = `Glucose` (capital G) and `steps` | not selected by the `glucose` filter; reported as unselected values |
| `app_glucose.csv` | value `LOW` | keep `raw_value`; value empty; `unparseable_numeric` |
| `app_glucose.csv` lines 8-9 | missing timestamp; missing participant ID | reject the rows, keep their content |
| `profile.csv` | two columns named `Time` | the second becomes `Time.1` (fingerstick time) |
| `profile.csv` | `Height ` with a trailing space; pounds and inches | resolve the alias; convert to kg and cm |
| `profile.csv` | LDL = 999 | documented error code: value set to empty, original kept, flagged |
| `profile.csv` line 4 | empty row | reject the row (`empty_row`) |
| `profile.csv` | weight `90 kg` (unit inside the value) | keep `raw_value`; value empty; `unparseable_numeric` (not reinterpreted) |
| device export | timezone not documented | localise with an explicit, registered assumption (`A-DEMO-TZ-01`) |

## How to use it

From the repository root:

```bash
env -u PYTHONPATH .venv/bin/pghd-ready ingest \
  --source    data/synthetic/wearable_demo/config/source.yaml \
  --concepts  configs/conditions/cardiometabolic.yaml \
  --data-root data/synthetic/wearable_demo/raw \
  --out       data/processed/wearable_demo
```

Then read `data/processed/wearable_demo/ingestion_report.md`. The output folder
is git-ignored and the run is reproducible: running it again regenerates the
same tables.

Expected result: **completed with warnings**, because the imperfections above
are intentional. The tests in `tests/test_ingest.py` and
`tests/test_wearable_demo.py` check the exact behaviour.

## Licence and provenance

Created by hand for this repository as synthetic test material. It contains no
data derived from CGMacros or any other real dataset.
