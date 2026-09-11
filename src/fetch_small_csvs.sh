#!/usr/bin/env bash
# Fetch ONLY the small CSVs from the competition — not the ~570 GB of DICOM.
#
# Why this exists: the CSVs are a few MB and they carry everything the report-labeling
# work needs (the 4,407 reports, the 58 gold label rows, the series metadata). You do not
# need the pixels to do Phase 0.
#
# This will NOT work from the Claude Code sandbox this repo was authored in — kaggle.com is
# blocked there at the egress-proxy level (see WORKING_NOTE.md §1.7). Run it somewhere with
# normal network access.
#
# Requires ~/.kaggle/kaggle.json (chmod 600) and `pip install kaggle`.
set -euo pipefail

COMP=rsna-knee-abnormality-detection
OUT=${1:-data}
mkdir -p "$OUT"

for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
  echo ">>> $f"
  kaggle competitions download -c "$COMP" -f "$f" -p "$OUT" --force || {
    echo "    failed: $f (may not exist under that name — run 'kaggle competitions files $COMP')"
    continue
  }
done

# Kaggle wraps single-file downloads in .zip when they are large enough.
for z in "$OUT"/*.zip; do
  [ -e "$z" ] || continue
  unzip -o "$z" -d "$OUT" && rm -f "$z"
done

echo
echo "Done. Contents of $OUT:"
ls -la "$OUT"
echo
echo "Total size: $(du -sh "$OUT" | cut -f1)  (expect a few MB, NOT gigabytes)"
