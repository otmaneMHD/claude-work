#!/usr/bin/env python3
"""
Phase 0 verification — convert the second-hand facts in WORKING_NOTE.md §3 into measurements.

WHY THIS EXISTS
---------------
The working note was written in a sandbox that could not reach kaggle.com (§1.2, §1.7). Every
number in §3 came from other teams' public repositories. They are good sources and they corroborate
each other (§3.9), but they are not our measurements and they are weeks stale.

This script re-measures all of them and prints a pass/fail table against the asserted values.
Run it FIRST, before building anything.

WHERE TO RUN
------------
A Kaggle **CPU** session (Settings -> Accelerator -> None). A CPU session draws nothing from the
30 h/week GPU quota. Expect ~10-30 minutes depending on --scan-mode.

    python phase0_verify.py                      # full header scan, all series
    python phase0_verify.py --scan-mode sample   # ~1200 series, ~2 min, good enough for most checks
    python phase0_verify.py --scan-mode none     # CSV checks only, seconds

PRIVACY / RULES NOTE
--------------------
This script never prints report text. Report columns are reduced to aggregate statistics only
(length, script/charset, presence of section headers). That is deliberate: WORKING_NOTE.md §1.6
flags Competition Rule 4.b, and keeping raw report text out of logs and notebook outputs costs
nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- asserted values
# Everything WORKING_NOTE.md §3 claims, so this script can grade itself.
# Tolerances are deliberately loose: we are checking "is the picture right", not "is it identical".
ASSERTED = {
    "train_studies":        (4407,   0.00),
    "train_series":         (24371,  0.00),
    "gold_studies":         (58,     0.00),
    "report_only_studies":  (4349,   0.00),
    "reports_present_frac": (1.00,   0.02),
    "n_labels":             (12,     0.00),
    "median_series_per_study": (5,   0.25),
    "median_slices_per_series": (30, 0.25),
    "fluid_fatsat_offdiag": (0,      0.00),   # they are claimed perfectly correlated
    "laterality_missing_frac": (0.507, 0.10),
    "bilateral_studies":    (25,     1.00),   # order-of-magnitude check only
    "pixelspacing_p50":     (0.312,  0.15),
    "pixelspacing_ratio_p99_p1": (5.14, 0.40),
    "fov_p1_mm":            (130,    0.10),
    "uncompressed_frac":    (1.00,   0.02),
    "patientid_repeats":    (0,      0.00),
}

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

CANDIDATE_ROOTS = [
    "/kaggle/input/rsna-knee-abnormality-detection",
    "./data", "../data", ".",
]

RESULTS: list[tuple] = []


def check(name: str, measured, note: str = "") -> None:
    """Grade one measurement against the asserted value and record it."""
    if name not in ASSERTED:
        RESULTS.append((name, measured, "-", "INFO", note))
        return
    expected, tol = ASSERTED[name]
    if measured is None:
        RESULTS.append((name, "n/a", expected, "SKIP", note))
        return
    if tol == 0.0:
        ok = measured == expected
    else:
        denom = abs(expected) if expected else 1.0
        ok = abs(measured - expected) / denom <= tol
    RESULTS.append((name, measured, expected, "PASS" if ok else "**FAIL**", note))


# --------------------------------------------------------------------------- locating the data
def find_root(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if (p / "train.csv").exists():
            return p
        sys.exit(f"no train.csv under {p}")
    for c in CANDIDATE_ROOTS:
        p = Path(c)
        if (p / "train.csv").exists():
            return p
    # last resort: anything under /kaggle/input that has a train.csv
    for p in Path("/kaggle/input").glob("*"):
        if (p / "train.csv").exists():
            return p
    sys.exit("could not locate the competition data. Pass --root explicitly.")


# --------------------------------------------------------------------------- CSV-level checks
def script_of(text: str) -> str:
    """Coarse writing-system bucket. A stand-in for language ID with no external deps.

    Deliberately crude: we only need enough resolution to confirm the corpus really is
    multilingual and to build the `language|manufacturer|model` fold key.
    """
    counts = Counter()
    for ch in text[:400]:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "CYRILLIC" in name:
            counts["cyrillic"] += 1
        elif "GREEK" in name:
            counts["greek"] += 1
        elif "LATIN" in name:
            counts["latin"] += 1
    return counts.most_common(1)[0][0] if counts else "unknown"


def csv_checks(root: Path) -> dict:
    print("\n" + "=" * 78)
    print("PART 1 — CSV-level checks")
    print("=" * 78)

    train = pd.read_csv(root / "train.csv")
    series = pd.read_csv(root / "train_series.csv")
    print(f"train.csv        {train.shape[0]:>7,} rows x {train.shape[1]} cols")
    print(f"train_series.csv {series.shape[0]:>7,} rows x {series.shape[1]} cols")
    print(f"train.csv columns: {list(train.columns)}")
    print(f"train_series.csv columns: {list(series.columns)}")

    check("train_studies", int(train["StudyInstanceUID"].nunique()))
    check("train_series", int(series["SeriesInstanceUID"].nunique()))

    # --- the 58 -----------------------------------------------------------------
    label_cols = [c for c in train.columns if c in LABELS]
    if not label_cols:  # tolerate renamed/normalised columns
        label_cols = [c for c in train.columns
                      if c not in ("StudyInstanceUID", "PatientID") and train[c].dropna().isin([0, 1]).all()
                      and train[c].notna().any()]
    check("n_labels", len(label_cols), f"found: {label_cols}")

    if label_cols:
        fully = train[label_cols].notna().all(axis=1)
        check("gold_studies", int(fully.sum()))
        check("report_only_studies", int((~fully).sum()))
        print("\nLabel prevalence among fully-labelled studies "
              "(NB: this subset is reportedly ~2x enriched — §3.2):")
        gold = train.loc[fully, label_cols]
        prev = pd.DataFrame({"n_pos": gold.sum().astype(int),
                             "rate": (gold.mean()).round(3)}).sort_values("n_pos", ascending=False)
        print(prev.to_string())
        # Every gold study reportedly has >=1 positive. That is a strong claim; test it.
        n_all_neg = int((gold.sum(axis=1) == 0).sum())
        check("gold_all_negative_studies", n_all_neg,
              "§3.2 claims every gold study has >=1 positive, i.e. this should be 0")

    # --- reports -----------------------------------------------------------------
    rep_col = next((c for c in train.columns if "report" in c.lower()), None)
    if rep_col:
        rep = train[rep_col].fillna("")
        check("reports_present_frac", round(float((rep.str.len() > 0).mean()), 4))
        wc = rep.str.split().str.len()
        print(f"\nReport length (words): p5={wc.quantile(.05):.0f} p50={wc.median():.0f} "
              f"p95={wc.quantile(.95):.0f} max={wc.max():.0f}")
        print(f"Reports under 50 words: {(wc < 50).mean():.1%}  (§3.3 warns this varies "
              f"0.6%-33.9% BY LANGUAGE and biases 'not mentioned')")
        scripts = rep.map(script_of).value_counts()
        print("\nWriting system (crude proxy for language — §3.3):")
        print(scripts.to_string())
        # §3.3 substitution artifact: numeric fragments replaced by the token 'intact'
        artifact = rep.str.contains(r"intact\d", case=False, regex=True, na=False)
        check("report_substitution_artifacts", int(artifact.sum()),
              "§3.3: strings like 'intact9xintact4cm' — clean before numeric parsing")
        hdr = rep.str.contains(r"^[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ \t]{3,}:", flags=re.M, regex=True, na=False)
        print(f"Reports with section headers: {hdr.mean():.1%} "
              f"(§3.3: ranges 1.6% German -> 99.8% Spanish)")
    else:
        print("\n!! no report column found in train.csv — check the column name")

    # --- the 6-slot structure -----------------------------------------------------
    if {"Fluid_Sensitive", "Fat_Suppression"}.issubset(series.columns):
        ct = pd.crosstab(series["Fluid_Sensitive"], series["Fat_Suppression"])
        print("\nFluid_Sensitive x Fat_Suppression (§3.4 claims a perfect diagonal):")
        print(ct.to_string())
        offdiag = int(ct.values.sum() - np.trace(ct.values)) if ct.shape[0] == ct.shape[1] else -1
        check("fluid_fatsat_offdiag", offdiag,
              "if 0, they are ONE bit -> 6 slots, not 12")

    if "Anatomical_Plane" in series.columns:
        series["slot"] = (series["Anatomical_Plane"].astype(str) + "|" +
                          np.where(series.get("Fluid_Sensitive", 0) == 1, "FLUID", "STRUCT"))
        n_studies = series["StudyInstanceUID"].nunique()
        cov = (series.groupby("slot")["StudyInstanceUID"].nunique() / n_studies).sort_values(ascending=False)
        print("\nSlot coverage — fraction of studies with >=1 series in that slot (§3.4):")
        print((cov * 100).round(1).to_string())

    spc = series.groupby("StudyInstanceUID").size()
    check("median_series_per_study", float(spc.median()))
    print(f"\nSeries per study: min={spc.min()} p50={spc.median():.0f} p95={spc.quantile(.95):.0f} max={spc.max()}")

    if "PatientID" in train.columns:
        check("patientid_repeats", int(len(train) - train["PatientID"].nunique()),
              "§4.5: zero repeats means folds need not group on patient")
    else:
        check("patientid_repeats", None, "PatientID absent from train.csv — read it from DICOM instead")

    return {"train": train, "series": series, "label_cols": label_cols}


# --------------------------------------------------------------------------- DICOM header scan
HEADER_TAGS = ["Laterality", "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
               "MagneticFieldStrength", "PixelSpacing", "Rows", "Columns", "SliceThickness",
               "SpacingBetweenSlices", "ImagePositionPatient", "ImageOrientationPatient",
               "SeriesDescription", "PatientSex", "PatientID", "InstanceNumber"]


def scan_one_series(args) -> dict | None:
    """Header-only read of the first slice of a series, plus a slice count. No pixels."""
    import pydicom

    series_dir, study_uid, series_uid = args
    try:
        files = sorted(os.listdir(series_dir))
        files = [f for f in files if f.endswith(".dcm")]
        if not files:
            return None
        ds = pydicom.dcmread(os.path.join(series_dir, files[0]), stop_before_pixels=True)
        out = {"StudyInstanceUID": study_uid, "SeriesInstanceUID": series_uid,
               "n_slices": len(files),
               "transfer_syntax": str(getattr(ds.file_meta, "TransferSyntaxUID", "")),
               "transfer_syntax_name": str(getattr(getattr(ds.file_meta, "TransferSyntaxUID", None), "name", ""))}
        for t in HEADER_TAGS:
            v = getattr(ds, t, None)
            if t == "PixelSpacing" and v is not None:
                # PixelSpacing is [row spacing, column spacing] — the two differ whenever
                # in-plane pixels are anisotropic, and conflating them skews x_centre.
                out["pixel_spacing"] = float(v[0])          # between rows (vertical)
                out["pixel_spacing_col"] = float(v[1]) if len(v) > 1 else float(v[0])
            elif t in ("ImagePositionPatient", "ImageOrientationPatient") and v is not None:
                out[t] = [float(x) for x in v]
            else:
                # blank strings are missing values — §5.6. Normalise them to None here.
                sv = None if v is None or str(v).strip() == "" else str(v)
                out[t] = sv
        return out
    except Exception as e:  # noqa: BLE001 — a scan must never die on one bad file
        return {"StudyInstanceUID": study_uid, "SeriesInstanceUID": series_uid, "error": repr(e)[:120]}


def geometric_side(row) -> str | None:
    """Side of the body the image centre sits on, from DICOM geometry.

    x_centre = IPP[0] + 0.5*Columns*ColSpacing*IOP[0] + 0.5*Rows*RowSpacing*IOP[3]

    IOP[0:3] is the direction of increasing COLUMN index, so it pairs with PixelSpacing[1];
    IOP[3:6] is the direction of increasing ROW index, so it pairs with PixelSpacing[0].
    Using one spacing for both is only correct for square pixels.

    DICOM patient coordinates are LPS, so +x = patient LEFT. §3.7 measures this rule at
    98.5% accuracy for |x| >= 20 mm.
    """
    ipp, iop = row.get("ImagePositionPatient"), row.get("ImageOrientationPatient")
    ps_row, rows_, cols = row.get("pixel_spacing"), row.get("Rows"), row.get("Columns")
    ps_col = row.get("pixel_spacing_col") or ps_row
    if not (isinstance(ipp, list) and isinstance(iop, list) and ps_row and rows_ and cols):
        return None
    try:
        x = (ipp[0] + 0.5 * float(cols) * ps_col * iop[0]
                    + 0.5 * float(rows_) * ps_row * iop[3])
    except Exception:  # noqa: BLE001
        return None
    if abs(x) < 20:
        return None           # inside the ambiguous band — §3.7
    return "L" if x > 0 else "R"


def dicom_scan(root: Path, series_df: pd.DataFrame, mode: str, workers: int) -> pd.DataFrame | None:
    if mode == "none":
        print("\n(skipping DICOM scan: --scan-mode none)")
        return None

    print("\n" + "=" * 78)
    print(f"PART 2 — DICOM header scan (mode={mode}, workers={workers})")
    print("=" * 78)

    base = root / "train_series"
    if not base.exists():
        print(f"!! {base} not found — skipping the DICOM scan")
        return None

    jobs = [(str(base / r.StudyInstanceUID / r.SeriesInstanceUID), r.StudyInstanceUID, r.SeriesInstanceUID)
            for r in series_df.itertuples()]
    if mode == "sample":
        rng = np.random.default_rng(0)
        jobs = [jobs[i] for i in rng.choice(len(jobs), size=min(1200, len(jobs)), replace=False)]
    print(f"scanning {len(jobs):,} series (headers only, stop_before_pixels=True)...")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for r in ex.map(scan_one_series, jobs, chunksize=32) if r is not None]
    dt = time.time() - t0
    df = pd.DataFrame(rows)
    n_err = int(df.get("error", pd.Series(dtype=object)).notna().sum()) if "error" in df else 0
    print(f"done in {dt/60:.1f} min  ({len(df):,} series, {n_err} read errors, "
          f"{dt/max(len(df),1)*1000:.1f} ms/series)")
    check("scan_read_errors", n_err, "§3.8 reports zero read errors on the full corpus")
    if "error" in df:
        df = df[df["error"].isna()].drop(columns=["error"])
    return df


def scan_checks(df: pd.DataFrame) -> None:
    # --- slices -----------------------------------------------------------------
    check("median_slices_per_series", float(df["n_slices"].median()))
    print(f"\nSlices per series: p5={df.n_slices.quantile(.05):.0f} p50={df.n_slices.median():.0f} "
          f"p95={df.n_slices.quantile(.95):.0f} p99={df.n_slices.quantile(.99):.0f} max={df.n_slices.max()}")

    # --- transfer syntax --------------------------------------------------------
    ts = df["transfer_syntax"].value_counts()
    print("\nTransfer syntax (§3.8 claims 100% Explicit VR Little Endian = 1.2.840.10008.1.2.1):")
    print(ts.to_string())
    check("uncompressed_frac", round(float((df["transfer_syntax"] == "1.2.840.10008.1.2.1").mean()), 4),
          "if <1.0, the cache build and inference runtime budgets change")

    # --- physical scale ---------------------------------------------------------
    ps = df["pixel_spacing"].dropna()
    if len(ps):
        check("pixelspacing_p50", round(float(ps.median()), 4))
        check("pixelspacing_ratio_p99_p1", round(float(ps.quantile(.99) / ps.quantile(.01)), 2),
              "§3.6: this is why a fixed-pixel resize is WRONG — crop to constant mm first")
        rows_ = pd.to_numeric(df["Rows"], errors="coerce")
        fov = (rows_ * df["pixel_spacing"]).dropna()
        if len(fov):
            check("fov_p1_mm", round(float(fov.quantile(.01)), 1),
                  "§3.6: CROP_MM=130 sits at the knee of the coverage curve")
            for mm in (120, 130, 140, 150, 160):
                print(f"   FOV >= {mm} mm covers {(fov >= mm).mean():6.2%} of series"
                      + ("   <-- CROP_MM candidate" if mm == 130 else ""))
            print("\n   mm/px at CROP_MM=130 — Nyquist needs <=0.5 mm for a ~1 mm meniscal tear:")
            for px in (224, 336, 448):
                pitch = 130 / px
                print(f"   {px:>3} px -> {pitch:.3f} mm/px  {'OK' if pitch <= 0.5 else 'TOO COARSE'}")

    # --- laterality -------------------------------------------------------------
    lat = df["Laterality"]  # scan_one_series already normalised "" -> None
    check("laterality_missing_frac", round(float(lat.isna().mean()), 4),
          "§5.6: blank strings ARE missing. Counting only NaN gives ~21%, the truth is ~51%")
    print(f"\nLaterality present on {lat.notna().mean():.1%} of series; values: "
          f"{lat.dropna().value_counts().to_dict()}")

    per_study = df.groupby("StudyInstanceUID")["Laterality"].agg(lambda s: set(s.dropna()))
    check("bilateral_studies", int((per_study.map(len) >= 2).sum()),
          "§3.7: ~25 (0.57%) — a footnote, not a workstream")
    check("studies_with_no_laterality", int((per_study.map(len) == 0).sum()),
          "these need the geometry fallback")

    # --- geometry fallback, validated against the tag ---------------------------
    df["geo_side"] = df.apply(geometric_side, axis=1)
    both = df[df["Laterality"].notna() & df["geo_side"].notna()]
    if len(both):
        acc = float((both["Laterality"].str[0].str.upper() == both["geo_side"]).mean())
        check("geometry_side_accuracy", round(acc, 4),
              f"§3.7 measures 98.5% at |x|>=20mm (n={len(both):,} here)")
        print("\nGeometry-vs-tag agreement by vendor (§3.7 — GE never carries the tag):")
        by_v = both.assign(agree=both["Laterality"].str[0].str.upper() == both["geo_side"]) \
                   .groupby(both["Manufacturer"].fillna("UNKNOWN"))["agree"].agg(["mean", "size"])
        print(by_v.sort_values("size", ascending=False).head(10).round(3).to_string())
    check("geometry_side_coverage", round(float(df["geo_side"].notna().mean()), 4),
          "§3.7: ~97.3% at the |x|>=20mm threshold")

    # --- fold grouping ----------------------------------------------------------
    print("\n" + "-" * 78)
    print("Fold grouping (§4.5 — random K-fold inflates AUC by 0.05-0.14)")
    print("-" * 78)
    scanner = (df["Manufacturer"].fillna("?") + "|" + df["ManufacturerModelName"].fillna("?"))
    study_scanner = df.assign(k=scanner).groupby("StudyInstanceUID")["k"].first()
    for name, key in [("manufacturer", df["Manufacturer"].fillna("?")),
                      ("manu|model", scanner)]:
        g = df.assign(k=key).groupby("StudyInstanceUID")["k"].first()
        vc = g.value_counts()
        print(f"  {name:<14} groups={len(vc):>4}  largest={vc.iloc[0]/len(g):6.1%}  "
              f"singletons={(vc == 1).sum():>4}  groups>=50={(vc >= 50).sum():>3}")
    print("  NB: add the report-language axis to get `language|manufacturer|model` "
          "(§4.5 measures 75 groups, max 5.8%) — that is the shipped scheme.")
    print("  WARNING: do NOT include ImagingFrequency in the key. It varies per SCAN, not per "
          "scanner, and produces 3,262 groups of which 2,668 are singletons (§4.5).")

    # --- PatientSex, absent from train.csv but present in DICOM (§3.5) ----------
    if "PatientSex" in df:
        check("patientsex_present_frac", round(float(df["PatientSex"].notna().mean()), 4),
              "§3.5: absent from train.csv but present in DICOM — and available at TEST time")

    # --- SeriesDescription usability (§3.5) -------------------------------------
    if "SeriesDescription" in df:
        sd = df["SeriesDescription"]
        dummy = sd.fillna("").str.contains("DummySeriesDesc", case=False)
        check("seriesdesc_placeholder_frac", round(float(dummy.mean()), 4),
              "§3.5: ~11.8% placeholder -> keep the CSV columns primary")


# --------------------------------------------------------------------------- report
def report(out_json: str | None) -> int:
    print("\n" + "=" * 78)
    print("VERIFICATION SUMMARY  —  measured vs WORKING_NOTE.md §3")
    print("=" * 78)
    w = max(len(r[0]) for r in RESULTS) + 2
    fails = 0
    for name, measured, expected, status, note in RESULTS:
        if status == "**FAIL**":
            fails += 1
        print(f"{name:<{w}} {str(measured):>12}  vs {str(expected):>10}  {status}")
        if note:
            print(f"{'':<{w}}   {note}")

    # Facts we ASSERT but never evaluated are NOT passes. With --scan-mode none the whole
    # DICOM half is silently absent from RESULTS, so "0 failures" must not read as "verified".
    evaluated = {r[0] for r in RESULTS}
    outstanding = [k for k in ASSERTED if k not in evaluated]
    outstanding += [r[0] for r in RESULTS if r[3] == "SKIP"]
    if outstanding:
        print("\nOUTSTANDING — asserted in §3, NOT measured by this run:")
        for k in outstanding:
            print(f"  - {k}")
        print("  Re-run with --scan-mode full (or sample) to close these.")

    print("\n" + "=" * 78)
    if fails == 0 and outstanding:
        print(f"NO FAILURES, BUT {len(outstanding)} OF {len(ASSERTED)} ASSERTED FACTS ARE UNMEASURED.")
        print("§3 is NOT yet [M] and Gate G1 is NOT passed. Close the outstanding list first.")
    elif fails == 0:
        print("ALL CHECKS PASSED — §3 is now [M] (measured). Proceed to Phase 1.")
    else:
        print(f"{fails} CHECK(S) FAILED.")
        print("A failure is INFORMATION, not an error in this script. It means the second-hand")
        print("fact was wrong or is stale. Update WORKING_NOTE.md §3 with the measured value,")
        print("re-read the decision it feeds in PLAN.md §4, and note it in the decision log.")
    print("=" * 78)

    if out_json:
        RESULTS.extend((k, "not measured", str(ASSERTED[k][0]), "OUTSTANDING", "")
                       for k in outstanding if k in ASSERTED)
        Path(out_json).write_text(json.dumps(
            [{"check": n, "measured": str(m), "expected": str(e), "status": s, "note": nt}
             for n, m, e, s, nt in RESULTS], indent=2))
        print(f"\nwrote {out_json}")
    return 1 if (fails or outstanding) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="competition data dir (auto-detected on Kaggle)")
    ap.add_argument("--scan-mode", choices=["full", "sample", "none"], default="full")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)))
    ap.add_argument("--out-json", default="phase0_results.json")
    ap.add_argument("--save-scan", default="series_headers.parquet",
                    help="the header scan is reused by every later stage — keep it")
    a = ap.parse_args()

    root = find_root(a.root)
    print(f"competition root: {root}")
    print(f"files: {sorted(p.name for p in root.glob('*.csv'))}")

    d = csv_checks(root)
    scan = dicom_scan(root, d["series"], a.scan_mode, a.workers)
    if scan is not None and len(scan):
        scan_checks(scan)
        if a.save_scan:
            try:
                scan.to_parquet(a.save_scan, index=False)
                print(f"\nsaved header scan -> {a.save_scan} "
                      f"({len(scan):,} rows). Phase 1 reuses this; do not re-scan.")
            except Exception as e:  # noqa: BLE001
                print(f"\ncould not write parquet ({e}); falling back to csv")
                scan.to_csv(a.save_scan.replace(".parquet", ".csv"), index=False)

    return report(a.out_json)


if __name__ == "__main__":
    sys.exit(main())
