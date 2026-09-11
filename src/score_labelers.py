#!/usr/bin/env python3
"""
Head-to-head: presence extraction vs severity grading, scored on the gold-labelled studies.

THIS IS THE PHASE-0 EXIT CRITERION (PLAN.md Phase 0, exit #3).

Bet A (WORKING_NOTE.md §8) says severity-graded soft targets beat binary presence extraction,
because the official rubric is severity-thresholded and the metric is rank-based. That is a
BET, not a belief. This script tries to falsify it.

    PASS  -> severity beats presence on >=8 of 12 labels. Train on severity targets.
    FAIL  -> fall back to presence labels, drop bet A to Tier 3, and move the GPU hours
             saved into Phase 3 (PLAN.md §5).

READ THE CAVEATS BEFORE READING THE NUMBER
------------------------------------------
The gold set is ~58 studies, reportedly ~2x ENRICHED in prevalence, and every one of them has
at least one positive (§3.2). At n=58 the standard error on an AUC is roughly +/-0.06.

  * You CANNOT resolve differences below ~0.02-0.03 here.
  * Read DIRECTION, not magnitude. If 9-11 of 12 labels move the same way, that is signal.
    One label improving is noise.
  * NEVER tune against this number. Another team iterated three times against the same 58 and
    had to state that their +0.046 was "partly in-sample" as a result (§4.3). Decide the
    design first, measure once, and write down what you predicted.
  * MCL reportedly has ~9 positives. Do not tune on MCL. Anything that "fixes" a label with 9
    positives is fitting noise (§5.4c).

Usage:
    python score_labelers.py --root /kaggle/input/rsna-knee-abnormality-detection
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from report_labeler import LABELS, label_presence, label_report  # noqa: E402


def auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    """Rank-based ROC-AUC (Mann-Whitney U). No sklearn dependency, handles ties correctly."""
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    m = ~(np.isnan(y_true) | np.isnan(score))
    y_true, score = y_true[m], score[m]
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    i = 0
    while i < len(sorted_scores):                 # average ranks within tie blocks
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_delta(y: dict, sev: dict, pres: dict, n_boot: int = 2000, seed: int = 0):
    """Bootstrap CI on the macro-AUC delta, resampling STUDIES (not labels)."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(y.values())))
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ds, dp = [], []
        for lab in LABELS:
            a = auc(y[lab][idx], sev[lab][idx])
            b = auc(y[lab][idx], pres[lab][idx])
            if a is not None and b is not None:
                ds.append(a)
                dp.append(b)
        if ds:
            deltas.append(np.mean(ds) - np.mean(dp))
    if not deltas:
        return None, None, None
    d = np.array(deltas)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input/rsna-knee-abnormality-detection")
    ap.add_argument("--out", default="weak_labels.csv",
                    help="write severity+confidence for ALL studies (the training targets)")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    root = Path(a.root)
    train = pd.read_csv(root / "train.csv")
    rep_col = next((c for c in train.columns if "report" in c.lower()), None)
    if rep_col is None:
        sys.exit("no report column found in train.csv")
    label_cols = [c for c in train.columns if c in LABELS]
    if len(label_cols) != 12:
        print(f"!! found {len(label_cols)} label columns, expected 12: {label_cols}")

    print(f"{len(train):,} studies; scoring both labelers over all reports...")
    reports = train[rep_col].fillna("").tolist()
    sev_rows = [label_report(t) for t in reports]
    pres_rows = [label_presence(t) for t in reports]

    # --- write the training targets for every study ------------------------------
    out = pd.DataFrame({"StudyInstanceUID": train["StudyInstanceUID"]})
    for lab in LABELS:
        out[f"sev::{lab}"] = [r[lab].severity for r in sev_rows]
        out[f"conf::{lab}"] = [r[lab].confidence for r in sev_rows]
        out[f"pres::{lab}"] = [r[lab] for r in pres_rows]
        out[f"state::{lab}"] = [r[lab].state for r in sev_rows]
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(out):,} rows) — these are the Phase-1 training targets\n")

    # --- diagnostics that catch the failure modes of §5.5 -------------------------
    print("Prediction spread per label — a COLLAPSE here (std ~0) is the §5.5 failure mode:")
    spread = pd.DataFrame({
        "std": [out[f"sev::{l}"].std() for l in LABELS],
        "n_distinct": [out[f"sev::{l}"].nunique() for l in LABELS],
        "mean": [out[f"sev::{l}"].mean() for l in LABELS],
    }, index=LABELS).round(3)
    print(spread.to_string())
    if (spread["n_distinct"] < 3).any():
        print("!! WARNING: a label has <3 distinct severity values. AUC cannot rank within a "
              "tie block — that label is effectively binary and bet A is not being tested on it.")
    print()

    print("State distribution — where each label's signal comes from:")
    states = pd.DataFrame({l: out[f"state::{l}"].value_counts() for l in LABELS}).T.fillna(0).astype(int)
    print(states.to_string())
    print()

    # --- the head-to-head on the gold studies -------------------------------------
    if not label_cols:
        print("no gold label columns present — cannot score. Exiting.")
        return 0
    gold_mask = train[label_cols].notna().all(axis=1)
    n_gold = int(gold_mask.sum())
    print("=" * 78)
    print(f"HEAD-TO-HEAD on {n_gold} gold studies")
    print("=" * 78)
    if n_gold < 20:
        print(f"!! only {n_gold} gold studies — far too few to read anything. Aborting.")
        return 0

    g = gold_mask.values
    y = {l: train.loc[g, l].values.astype(float) for l in label_cols}
    sev = {l: out.loc[g, f"sev::{l}"].values for l in label_cols}
    pres = {l: out.loc[g, f"pres::{l}"].values for l in label_cols}

    rows = []
    for lab in label_cols:
        n_pos = int((y[lab] == 1).sum())
        as_, ap_ = auc(y[lab], sev[lab]), auc(y[lab], pres[lab])
        rows.append({"label": lab, "n_pos": n_pos,
                     "presence": ap_, "severity": as_,
                     "delta": (as_ - ap_) if (as_ is not None and ap_ is not None) else None})
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))

    valid = res.dropna(subset=["delta"])
    mp, ms = valid["presence"].mean(), valid["severity"].mean()
    improved = int((valid["delta"] > 0).sum())
    n_valid = len(valid)
    print(f"\nmacro presence = {mp:.4f}")
    print(f"macro severity = {ms:.4f}")
    print(f"delta          = {ms - mp:+.4f}   improved on {improved}/{n_valid} labels")

    lo, hi, pgt = bootstrap_delta(y, sev, pres, a.n_boot)
    if lo is not None:
        print(f"bootstrap 95% CI on delta: [{lo:+.4f}, {hi:+.4f}]   P(delta>0) = {pgt:.1%}")

    # --- rubric-family split, the §4.4 prediction ---------------------------------
    from report_labeler import MAGNITUDE_LABELS
    for fam, sel in [("magnitude (how much?)", valid["label"].isin(MAGNITUDE_LABELS)),
                     ("categorical (what kind?)", ~valid["label"].isin(MAGNITUDE_LABELS))]:
        sub = valid[sel]
        if len(sub):
            print(f"  {fam:<26} presence={sub['presence'].mean():.4f}  "
                  f"severity={sub['severity'].mean():.4f}  "
                  f"delta={sub['severity'].mean()-sub['presence'].mean():+.4f}  "
                  f"improved {int((sub['delta']>0).sum())}/{len(sub)}")

    print("\n" + "=" * 78)
    if improved >= 8:
        print(f"BET A SURVIVES — severity improved {improved}/{n_valid} labels.")
        print("Train Phase 1 on the severity targets in", a.out)
    else:
        print(f"BET A FALSIFIED — severity improved only {improved}/{n_valid} labels.")
        print("Fall back to presence labels, drop bet A to Tier 3, and move the saved GPU")
        print("hours into Phase 3 (PLAN.md §5 'What would make us change course').")
    print(f"\nReminder: n={n_gold}, ~2x enriched, SE(AUC) ~ +/-0.06. Read DIRECTION, not")
    print("magnitude. Do not tune against this number — you only get to use it honestly once.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
