#!/usr/bin/env python3
"""
Site-grouped cross-validation folds.

WHY (WORKING_NOTE.md §4.5, §8 bet C)
------------------------------------
Random K-fold on this dataset inflates AUC by a MEASURED +0.053 from DICOM headers alone, and
by +0.136 once a CNN sees pixels. The model learns which scanner took the picture, not what is
wrong with the knee. Every architectural decision made on random-fold CV is a coin flip.

THE GROUPING KEY: `language | manufacturer | model`
--------------------------------------------------
Language is an excellent site proxy — Cyrillic reports are 100% Philips; Dutch, German and
Greek are 100% Siemens. Those are single institutions showing through. Measured group counts
over 4,407 studies (§4.5):

    manufacturer           7 groups, largest 44.3%   too coarse
    manu|model            46 groups, largest 16.8%   better
    language|manu|model   75 groups, largest  5.8%   <-- shipped

DO NOT include ImagingFrequency. It varies per SCAN (63.685238 vs 63.685256), not per scanner,
and explodes the key into 3,262 groups of which 2,668 are singletons.

REPORT BOTH FOLD SCHEMES, EVERY RUN
-----------------------------------
Train and test come from the SAME 16 sites, so the test set is not an unseen-site holdout.
That makes grouped CV PESSIMISTIC and random CV OPTIMISTIC; neither is the target. The GAP
between them measures how much the model leans on site rather than anatomy — treat it as a
first-class metric and drive it down (§8 D2).
"""
from __future__ import annotations

import unicodedata
from collections import Counter

import numpy as np
import pandas as pd


def script_bucket(text: str) -> str:
    """Coarse writing-system bucket: a dependency-free stand-in for language ID.

    Enough resolution to separate the Cyrillic / Greek / Latin institutions. For a finer split
    inside Latin, add a stopword-based detector or `langdetect` — but note that anything you
    add here changes the fold boundaries, so freeze it before you start comparing runs.
    """
    counts: Counter = Counter()
    for ch in str(text)[:400]:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for s in ("CYRILLIC", "GREEK", "LATIN"):
            if s in name:
                counts[s.lower()] += 1
                break
    return counts.most_common(1)[0][0] if counts else "unknown"


def build_group_key(train: pd.DataFrame, series_headers: pd.DataFrame,
                    report_col: str = "report") -> pd.Series:
    """One group label per study: `language|manufacturer|model`.

    train           : train.csv (needs StudyInstanceUID + the report column)
    series_headers  : the parquet written by phase0_verify.py (needs StudyInstanceUID,
                      Manufacturer, ManufacturerModelName)
    """
    scanner = (series_headers.groupby("StudyInstanceUID")
               .agg(manu=("Manufacturer", "first"), model=("ManufacturerModelName", "first")))
    lang = train.set_index("StudyInstanceUID")[report_col].fillna("").map(script_bucket)
    df = pd.DataFrame({"lang": lang}).join(scanner, how="left")
    return (df["lang"].fillna("unknown") + "|" +
            df["manu"].fillna("?").astype(str).str.strip().str.upper() + "|" +
            df["model"].fillna("?").astype(str).str.strip().str.upper())


def grouped_folds(groups: pd.Series, n_splits: int = 5, seed: int = 0) -> pd.Series:
    """Assign folds so that no group is split across folds.

    Greedy largest-group-first bin packing — it keeps folds far more even than sklearn's
    GroupKFold when a handful of groups dominate, which is exactly our situation.
    """
    sizes = groups.value_counts()
    rng = np.random.default_rng(seed)
    order = sizes.index[np.lexsort((rng.random(len(sizes)), -sizes.values))]
    load = np.zeros(n_splits, dtype=int)
    assign: dict[str, int] = {}
    for g in order:
        f = int(np.argmin(load))
        assign[g] = f
        load[f] += int(sizes[g])
    return groups.map(assign).astype(int)


def random_folds(index: pd.Index, n_splits: int = 5, seed: int = 0) -> pd.Series:
    """The OPTIMISTIC control. Always report alongside grouped — the gap is the diagnostic."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, n_splits, len(index)), index=index)


def fold_report(groups: pd.Series, folds: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    rows = []
    for f in range(n_splits):
        m = folds == f
        rows.append({"fold": f, "n_studies": int(m.sum()),
                     "pct": round(100 * m.mean(), 1),
                     "n_groups": int(groups[m].nunique())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Self-test on synthetic groups shaped like the real distribution (§4.5: 75 groups,
    # largest 5.8%). Verifies no group leaks across folds and that folds stay balanced.
    rng = np.random.default_rng(0)
    sizes = np.clip((rng.pareto(1.6, 75) + 1) * 25, 3, 260).astype(int)
    g = pd.Series(np.concatenate([[f"g{i}"] * s for i, s in enumerate(sizes)]))
    g.index = [f"s{i}" for i in range(len(g))]
    f = grouped_folds(g, 5)

    leaked = [grp for grp, sub in f.groupby(g) if sub.nunique() > 1]
    rep = fold_report(g, f)
    print(rep.to_string(index=False))
    print(f"\ntotal studies {len(g):,}  groups {g.nunique()}  "
          f"largest group {g.value_counts().iloc[0] / len(g):.1%}")
    assert not leaked, f"groups leaked across folds: {leaked}"
    spread = rep.n_studies.max() / rep.n_studies.min()
    print(f"fold size spread (max/min) = {spread:.2f}")
    assert spread < 1.35, "folds too uneven"
    print("\nOK — no group spans two folds, folds balanced within 35%")
