# Pipeline scaffolding

Written to be pasted into Kaggle notebooks. Nothing here has run against real competition data —
this sandbox could not reach it (WORKING_NOTE.md §1.7) — but **everything has been smoke-tested
against a synthetic dataset shaped like the real one**, and the labeler and the AUC are unit-tested.

| File | What it does | Where to run | Tested |
|---|---|---|---|
| `phase0_verify.py` | Re-measures every [S] fact in WORKING_NOTE.md §3 and prints a pass/fail table. **Run this first.** Also writes `series_headers.parquet`, reused by every later stage | Kaggle **CPU** session (zero GPU quota), ~10–30 min | ✅ end-to-end on a synthetic 60-study / 300-series / 9,576-file dataset |
| `report_labeler.py` | The Tier-1 bet: multilingual severity-graded weak labels. Negation-first, two extractors, `severity` (rank) and `confidence` (uncertainty) kept separate | anywhere, seconds | ✅ 20 unit tests across EN/ES/DE/TR/FR, all passing |
| `score_labelers.py` | **The Phase-0 exit criterion.** Presence vs severity head-to-head on the gold studies, with a bootstrap CI, and the collapse/spread diagnostics | anywhere, seconds | ✅ smoke-tested; its AUC matches `sklearn.roc_auc_score` exactly over 500 tie-heavy cases |
| `folds.py` | Site-grouped CV on `language\|manufacturer\|model`, plus the random control | anywhere | ✅ self-test: no group spans two folds, folds balanced within 35 % |
| `fetch_small_csvs.sh` | Pulls **only the CSVs** (a few MB), not the 570 GB. Needs network access to Kaggle | any unblocked machine | — |

## Order of operations

```bash
# 1. Kaggle CPU session — verify the facts, build the header scan
python phase0_verify.py --scan-mode full
#    -> phase0_results.json, series_headers.parquet

# 2. anywhere — build weak labels and test the Tier-1 bet
python report_labeler.py                    # unit tests must pass first
python score_labelers.py --root /kaggle/input/rsna-knee-abnormality-detection
#    -> weak_labels.csv, and a PASS/FAIL verdict on bet A

# 3. anywhere — folds
python folds.py                             # self-test
```

Only after all three pass does anything touch a GPU. See [../PLAN.md](../PLAN.md) Phase 0.

## Not yet written

The cache builder, dataset/model/training loop and inference notebook are Phase 1 and deliberately
left until §11 has been answered with real data. Writing them now would bake in the [S] assumptions
this whole repo is trying to verify — for example the cache resolution, which depends on whether
the 336 px hypothesis (§8 D1) survives.
