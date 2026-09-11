# RSNA 2026 Knee Abnormality Detection — exhaustive working note

**Compiled:** 2026-09-11 · **Author:** Claude (Claude Code, remote sandbox session)
**Competition state at time of writing:** open since 2026-07-30, final submission **2026-10-22** — **41 days remain**
**Work state:** research complete · **no data downloaded** · **nothing trained** · no submission made

---

## Table of contents

- [§0 Handoff contract](#0-handoff-contract)
- [§1 Session audit — what was done, what was blocked](#1-session-audit)
- [§2 The competition](#2-the-competition)
- [§3 The data](#3-the-data)
- [§4 What kind of problem this actually is](#4-what-kind-of-problem-this-actually-is)
- [§5 The public landscape](#5-the-public-landscape)
- [§6 Prior competitions surveyed, and what transfers](#6-prior-competitions-surveyed)
- [§7 Literature, anatomy, and external assets](#7-literature-anatomy-and-external-assets)
- [§8 Technical thesis — the ranked bets](#8-technical-thesis)
- [§9 The Efficiency Track](#9-the-efficiency-track)
- [§10 Risk register / trap list](#10-risk-register)
- [§11 Open questions — verify these first](#11-open-questions)
- [§12 Provenance and confidence ledger](#12-provenance-and-confidence-ledger)

---

## Confidence legend

Used on every factual claim in this document. **Do not skip this.**

| Tag | Meaning |
|---|---|
| **[M]** | **Measured by us** in this session. Reproducible from the commands in §1. |
| **[C]** | **Corroborated** — asserted independently by two or more sources that did not copy each other. |
| **[S]** | **Second-hand single source** — one public document asserts it; we could not check it. |
| **[I]** | **Inference** — our reasoning from [M]/[C]/[S] facts. Flagged where the chain is long. |
| **[?]** | **Unverified / open.** Listed in §11 with the command that would settle it. |

The single most important thing a successor can do with this document is **convert [S] facts to
[M] facts** in the first Kaggle session. §11 is that checklist.

---

## 0. Handoff contract

### 0.1 One paragraph

Predict **12 binary abnormality findings per knee MRI study** from multi-plane MRI, scored by
**macro-averaged ROC-AUC**, as a Kaggle **code competition** (notebook, internet off, ≤9 h). The
training set is ~4,400 studies and ~570 GB of DICOM — but **only 58 studies carry the 12 official
labels**. The other ~4,349 come with a free-text radiology report, in **9–12 languages**, written
by a different observer under a different standard than the ground truth. So this is **not
primarily a computer-vision competition; it is a weak-supervision competition** in which the team
that best converts multilingual reports into training targets sets the ceiling for every vision
model trained downstream. Two further facts shape everything: the official labels are
**severity-thresholded** ("on the fence" → negative), so presence-extraction from reports is
systematically wrong in a *correctable, monotone* direction; and **random K-fold cross-validation
inflates scores by ~0.05–0.14 through scanner/site memorisation**, so grouped CV is mandatory.

### 0.2 What a successor must NOT redo

| Already done — do not repeat | Where it lives |
|---|---|
| Establishing the task, metric, rules, timeline, prize structure | §2 |
| Establishing data scale, label sparsity, language mix, series structure | §3 |
| Establishing that this is a weak-supervision problem and why | §4 |
| Surveying the public LB tiers and the public baseline's design | §5 |
| Selecting and mining prior competitions for transferable method | §6 |
| Knee-MRI plane/anatomy evidence, DINOv2-in-medical evidence, dataset licensing status | §7 |
| The efficiency-metric arithmetic and what it implies | §9 |
| The trap list | §10 |

### 0.3 What a successor MUST do, in order

1. **Open a Kaggle CPU session** (costs no GPU quota) and run the verification checklist in §11.
   This converts ~20 [S] facts to [M] in about 30 minutes.
2. **Build the report labeler** (§8 bet A) and score it on the 58 gold studies. Pure text, no GPU.
3. **Build the pixel cache** on a CPU session (§PLAN Phase 1).
4. Only then train anything.

### 0.4 The honest caveat that governs this whole document

**No competition data was downloaded or inspected in this session.** The egress policy blocked
`kaggle.com` (§1.2). Every number in §3 is [S] or [C] from other teams' public work. They are
*good* sources — two independent repositories agree on the load-bearing numbers (§3.9) — but they
are not our measurements, they are ~5 weeks stale, and one of them is a **direct competitor** whose
strategic framing may be self-serving or simply wrong. Treat §3 as a very well-informed prior, not
as ground truth, and spend the first 30 minutes of Kaggle access on §11.

---

## 1. Session audit

Full trail of what this session did, so nobody repeats the dead ends.

### 1.1 Environment — measured [M]

```
Working dir     /home/user/claude-work (git repo, branch claude/pensive-euler-lt2rpc)
Platform        Linux 6.18.44-fc-v24, Python 3.11.15, pip 24.0
CPU             4 cores
RAM             15 GB total
Disk            ~30 GB writable allowance   <-- NOT the 252 GB that `df` reports
GPU             none (nvidia-smi absent)
Kaggle creds    none (~/.kaggle absent, no KAGGLE_* env vars)
kaggle CLI      not installed
```

Commands used: `df -h`, `free -g`, `nproc`, `nvidia-smi`, `ls ~/.kaggle`, `env | grep -i kaggle`.

**Consequence:** even with perfect network access and credentials, this box **cannot hold the
dataset** (~570 GB against a 30 GB allowance) and **cannot train** (no GPU). Downloading was never
going to be possible here. This is not a configuration problem to fix; it is the shape of the
sandbox. See §1.4 for what to do instead.

### 1.2 Network — measured [M]

Outbound HTTPS goes through a policy-enforcing egress proxy. Probed 19 hosts with
`curl -o /dev/null -w "%{http_code}"`:

| Reachable | Blocked (403 at CONNECT, or no route) |
|---|---|
| `github.com`, `raw.githubusercontent.com` | **`kaggle.com`, `www.kaggle.com`, `api.kaggle.com`** |
| `pypi.org`, `files.pythonhosted.org` | `arxiv.org`, `huggingface.co`, `pubmed.ncbi.nlm.nih.gov` |
| `storage.googleapis.com` | `openreview.net`, `paperswithcode.com`, `zenodo.org`, `physionet.org` |
| the `WebSearch` tool (separate path, works) | `google.com`, `duckduckgo.com`, `rsna.org`, `auntminnie.com` |

The proxy's own status endpoint recorded the denial verbatim:

```json
"recentRelayFailures": [
  { "kind": "connect_rejected",
    "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
    "host": "www.kaggle.com:443" } ]
```

`WebFetch` against `kaggle.com`, `rsna.org` and `auntminnie.com` returned
`{"error_type":"EGRESS_BLOCKED"}`. The proxy README is explicit: *"Do not retry or route around it
— report the blocked host."* We did not attempt to circumvent it.

**So: the competition page, the data page, the rules page, the discussion forum, the leaderboard,
the public notebooks and every winner writeup hosted on Kaggle were all unreadable directly.**

### 1.3 What was reachable, and what it gave us

| Channel | Status | What it delivered |
|---|---|---|
| `WebSearch` | works | Summarises Kaggle pages second-hand. Gave metric, timeline, prizes, data layout, LB facts |
| `git clone` from GitHub | works | **The highest-value channel.** Full repos, full docs, full source |
| `WebFetch` | works only on non-blocked hosts | Little use here — most radiology/news hosts are blocked |

**The find that rescued this session:** two public GitHub repositories from teams already competing
in this exact competition, both containing detailed written analysis of the real data:

1. **`homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection`** — 1,315 lines of strategy docs
   (`STRATEGY.md`, `FINDINGS.md`, `EXPERIMENTS.md`, `ROADMAP.md`, `PLATFORM.md`,
   `RESEARCH_AGENDA.md`, `DAY1.md`), a 586-line multilingual report labeler, EDA scripts, Kaggle
   kernel sources, and an experiment log with real training results. **Contains a measured
   header-only scan of 819,078 DICOM files.** Dated 2026-08-08 (competition day 9).
2. **`JunhaoLiXD/RSNA_Knee_Abnormality_Detection`** — a versioned 2.5D baseline (V01→V04) with
   Kaggle public scores logged per version, and an independent statement of the dataset counts.

These two teams did not copy each other (different code, different framing, different label
schemes) yet agree on the load-bearing numbers — see the corroboration table in §3.9.

Also cloned, for prior-art method: `MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution`
(7th/1147, RSNA 2025), `Nischaydnk/RSNA-2023-1st-place-solution`, `TheoViel/kaggle_rsna_abdominal_trauma`
(2nd, RSNA 2023), `dangnh0611/kaggle_rsna_breast_cancer` (1st, RSNA 2023 mammography).

### 1.4 The download question, answered plainly

**You asked me to download the 100+ GB competition dataset. I could not, and no configuration
change available to me would have made it possible.** Three independent blockers, any one of which
is fatal:

1. **`kaggle.com` is blocked by the organisation's egress policy** — a 403 at the proxy's CONNECT,
   not a credential or TLS problem.
2. **No Kaggle credentials** exist in this session.
3. **~30 GB of writable disk** against a ~570 GB dataset, and **no GPU** to use it with.

Even in a sandbox with open network and credentials, downloading it *here* would be the wrong
move. The dataset is already mounted, free, at `/kaggle/input/rsna-knee-abnormality-detection/`
inside any Kaggle notebook, and **this is a code competition — the final submission must be a
Kaggle notebook regardless.** Anything built elsewhere has to come home to Kaggle to score. So the
correct platform is Kaggle, and this box's job is what it just did: research, synthesis, planning,
and writing the code that will run there. See [PLAN.md](PLAN.md) §Platform.

### 1.5 Chronological audit trail of this session

| # | Action | Outcome |
|---|---|---|
| 1 | Inspected working dir, git state, disk, RAM, CPU, GPU, Kaggle creds | §1.1 — no GPU, 30 GB, no creds |
| 2 | Probed 19 hosts through the egress proxy; read proxy status + README | §1.2 — kaggle.com blocked |
| 3 | `WebFetch` competition page, RSNA challenge page, AuntMinnie article | All `EGRESS_BLOCKED` |
| 4 | `WebSearch` × 4 on competition identity, metric, data layout, timeline | Confirmed §2 facts |
| 5 | Found two competitor GitHub repos in search results; cloned both | The intel break |
| 6 | Read 1,315 lines of strategy docs + 586-line labeler + notebook markdown | §3, §4, §5, §8 |
| 7 | `WebSearch` × 8 on prior RSNA competitions and winner solutions | §6 |
| 8 | Cloned 4 winner-solution repos; read READMEs and pipeline descriptions | §6 |
| 9 | `WebSearch` × 6 on knee-MRI anatomy, MRNet, DINOv2-in-medical, datasets, report labeling | §7 |
| 10 | User supplied `kaggle.json`; installed CLI and retried | Still 403 at CONNECT — §1.7 |
| 11 | Wrote this note, [PLAN.md](PLAN.md), [docs/SOURCES.md](docs/SOURCES.md), `src/` scaffolding | this repo |
| 12 | Built a synthetic dataset and smoke-tested every script; unit-tested the labeler and the AUC | §1.8 — all green, one real bug found and fixed |

**Nothing was trained. No submission was made. No competition data was touched.**

### 1.6 A rules note about report text and hosted models

The competitor repo flags Competition Rule 4.b (Data Security) as plausibly forbidding sending
Competition Data to any non-participating party, which would include a hosted LLM API — and
discloses that they themselves printed ~10 report excerpts into a hosted assistant context before
realising. **[S], host has not ruled [?]**

**This session never handled any competition data at all**, report text included, because it never
had access to any. That exposure is zero here. But the constraint binds the *next* phase, and it is
a real one:

> **Do not paste report text into any hosted model — this assistant included — and do not call a
> commercial LLM API on report text.** Use open-weights multilingual models (Qwen3, Gemma,
> multilingual-e5) run locally or inside a Kaggle notebook. Internet-off applies only to the
> *submission* notebook, so offline label generation during development is unrestricted.

This costs nothing: open-weights multilingual models are fully adequate for report labeling, and
the rule-based path (§8 bet A) needs no model at all.


### 1.7 Kaggle credentials were supplied mid-session — and still do not help here

The user supplied a valid `kaggle.json` during this session. It was installed to
`~/.kaggle/kaggle.json` (mode 600, outside the repo, never committed) and the Kaggle CLI 2.2.4 was
installed. **The API still cannot reach Kaggle from this sandbox.** Measured **[M]**:

```
$ kaggle competitions files rsna-knee-abnormality-detection
HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded
  (Caused by ProxyError('Unable to connect to proxy',
   OSError('Tunnel connection failed: 403 Forbidden')))
```

Re-probed with the credentials in place: `api.kaggle.com`, `www.kaggle.com` and
`kaggleusercontent.com` all return **403 at the CONNECT stage** — that is the egress policy
refusing to open the tunnel, *before* any Kaggle authentication happens. The credentials are not
the problem and cannot be the solution. Per the proxy README, this is reported, not worked around.

**What this changes for the plan: nothing structural, one thing practically.** It was never the
570 GB that mattered from this box — it was the **small CSVs** (`train.csv` ≈ 5 MB with the reports
and the 58 gold labels, `train_series.csv`, `sample_submission.csv`). Those would have converted
most of §3 from [S] to [M] and let the report labeler be built and scored here. They cannot be
fetched from this sandbox.

So the labeler work moves to wherever the user has Kaggle access. To make that a paste-and-run job
rather than a re-derivation, this repo ships:

- `src/phase0_verify.py` — the §11 verification checklist as one script. Run on a **Kaggle CPU
  session** (costs zero GPU quota); it prints every number §3 asserts, marked pass/fail against the
  [S] value, so [S] becomes [M] in one run.
- `src/fetch_small_csvs.sh` — uses the supplied credentials to pull **only** the CSVs (a few MB, not
  570 GB) on any machine whose network permits it.

**Handling note:** the credential file was written with mode 600 outside the repository and is
covered by `.gitignore`. It is not in git history. Rotate it at
<https://www.kaggle.com/settings> if you would rather not have it sitting in a sandbox.

### 1.8 What was built and validated in this session — [M]

The sandbox could not reach the data, but it could run code. Everything in `src/` was
**smoke-tested against a synthetic dataset built to the shape §3 describes** (60 studies, 300
series, 9,576 DICOM files, 5 slots, 3 vendors, mixed pixel spacings, GE deliberately given blank
`Laterality` strings, reports in EN/ES/DE/TR).

| Artifact | Validation | Result |
|---|---|---|
| `src/phase0_verify.py` | Full run against the synthetic set | ✅ every check fired correctly. The 7 "FAIL"s were the synthetic set being 60 studies not 4,407 — the *mechanisms* (label detection, gold counting, fluid/fat-sat cross-tab, slot coverage, transfer-syntax census, FOV/Nyquist table, blank-string laterality handling, geometry sign rule, vendor breakdown, fold grouping, parquet write) all worked |
| `src/report_labeler.py` | 20 unit tests across English, Spanish, German, Turkish, French | ✅ all pass |
| — negation-first ordering | `"Medial meniscus: no tear"` → 0.01, in 5 languages | ✅ the §5.6 bug cannot recur |
| — heading attachment | `"Fractures :\nAucune."` → 0.01 | ✅ |
| — severity ladder | effusion `[0.01, 0.05, 0.15, 0.62, 0.90]` strictly increasing, 5 distinct values | ✅ |
| — rank invariant | negated 0.01 < silent 0.03 < mild 0.15 | ✅ the §5.5 inversion cannot recur |
| — rubric cut | "small effusion" → 0.15, "moderate" → 0.62; Outerbridge 2 → 0.15, 4 → 0.90 | ✅ |
| `src/score_labelers.py` — AUC | Hand-rolled Mann-Whitney vs `sklearn.roc_auc_score` over **500 random tie-heavy cases** | ✅ exact match to 1e-9 |
| `src/folds.py` | Self-test on a Pareto-shaped group distribution | ✅ no group spans two folds; folds balanced to 1.00×. Incidentally reproduced 75 groups / largest 5.9 % — almost exactly the real reported 75 / 5.8 % |

**One real bug was found and fixed by these tests**, and it is instructive because it is the same
*class* of bug as the two documented in §5.6: `"partial tear of the ACL"` scored **0.92** instead of
0.68, because the clause matched both the `partial tear` tier and the bare `\btear\b` in the
`complete` tier, and `max()` promoted it. That silently collapses the exact distinction the ACL and
meniscus rubrics turn on — **>50 % fibre disruption**, **signal reaching the surface**. Fixed with an
explicit partial/low-grade cap. **Ordering and specificity bugs in these extractors are invisible
without tests and expensive with them.** Keep the test suite green.


---

## 2. The competition

### 2.1 Identity and framing

- **Host:** RSNA (Radiological Society of North America), on Kaggle. **[C]**
- RSNA's **first** AI challenge on **musculoskeletal MRI**, and its **first to combine medical
  images with radiology report text** — and the reports are **multilingual**. **[C]**
- Announced Aug 2026; winners recognised in the AI Theater at RSNA 2026 (Nov 29 – Dec 3, McCormick
  Place, Chicago). **[C]**
- Framed by Kaggle/RSNA around osteoarthritis burden: *"Over 600M people globally live with
  osteoarthritis, and the knee is the most commonly affected joint."* **[S]**

### 2.2 Task

Per **study** (`StudyInstanceUID`), predict the probability of each of **12 binary findings**. A
study is one knee MRI examination containing several series in different planes and sequences. It
is **multilabel** — one study can be positive for several findings simultaneously. **[C]**

### 2.3 The 12 labels **[C — independently listed by both competitor repos]**

| # | Label | Clinical meaning |
|---|---|---|
| 1 | `ACL` | Anterior cruciate ligament abnormality |
| 2 | `MCL` | Medial collateral ligament abnormality |
| 3 | `Medial Meniscus` | Medial meniscus abnormality |
| 4 | `Lateral Meniscus` | Lateral meniscus abnormality |
| 5 | `Medial OA` | Medial tibiofemoral osteoarthritis |
| 6 | `Lateral OA` | Lateral tibiofemoral osteoarthritis |
| 7 | `PF OA` | Patellofemoral osteoarthritis |
| 8 | `Effusion` | Joint effusion |
| 9 | `Synovitis` | Synovitis |
| 10 | `Baker's` | Baker / popliteal cyst |
| 11 | `Contusion` | Bone contusion |
| 12 | `Fracture` | Fracture |

**Five of the twelve are side-specific** (`Medial Meniscus`, `Lateral Meniscus`, `Medial OA`,
`Lateral OA`, `MCL`). This single fact drives two hard constraints — laterality canonicalisation
and a ban on horizontal flip augmentation. See §10.

### 2.4 The official label criteria — the most under-read part of this competition **[S]**

Sourced to the host (Po-Hao "Howard" Chen) in the pinned overview and Q&A discussions. **Every one
of the twelve is severity-thresholded:**

| Label | Positive requires | Explicitly **negative** |
|---|---|---|
| ACL | *High-grade* partial or full-thickness tear — complete discontinuity or **>50 % of fibres** disrupted | Mild signal change, degeneration, thickening **without discontinuity** |
| MCL | High-grade partial or complete **acute** tear | Low-grade sprains; chronic / remote stress change |
| Meniscus (each) | Abnormal signal **definitely contacting the meniscal surface on ≥2 images**, or morphologic abnormality | Intrasubstance degeneration **not reaching the surface** |
| OA (each of 3) | **Moderate or large area (≥ ~1 cm)** of **high-grade** cartilage loss (**>50 % of thickness**) | Anything smaller or lower-grade |
| Effusion | **Moderate or large** fluid distending the joint | Trace / small |
| Baker's | **Moderate or large** fluid collection | Small |
| Contusion | Marrow-edema-like signal from impact, **without a discrete fracture line** | — |
| Fracture | **Acute cortical break or fracture line** | — |

And the governing rule:

> *"In each case, ambiguous or borderline findings ('on the fence') were graded as negative to
> favour specificity."*

**Read that again before designing any label extractor.** The ground truth does not ask *"is the
finding mentioned?"* — it asks *"is the finding severe and unambiguous?"* A report saying "small
joint effusion", "mild chondropathy", "grade 2b intrasubstance meniscal signal" or "low-grade MCL
sprain" is **correctly labelled 0**. Teams posting about "label errors" are mostly looking at the
rubric working as designed. §8 bet A is built entirely on this.

### 2.5 How ground truth was produced **[S]**

**Two subspecialty MSK radiologists read the images; a third adjudicates.** The host confirmed, in
answer to direct questions:

> **Q:** Were the labels assigned independently from the MRI images, rather than extracted from the
> reports? — **A: Yes.**
> **Q:** If image interpretation and report text disagree, should the image-derived label be
> considered authoritative? — **A: Yes.** *Note that only a small sample of provided data contains
> both. It is intended to help participants surface this conclusion.*

The reports, by contrast, are original clinical reads by a **single signing radiologist** at the
originating site, in routine care. **These are two different measurement instruments**, and the
host says the discrepancy is "plausible and expected… the image-based labels use multiple readers
with stricter image-based thresholds."

An independent audit by another team on 20 of the 58 dual-labelled studies reportedly found
report-derived labels agree with the official labels only **82.5 %** of the time (PPV 73 %, recall
80 %). **[S — single source, small n, treat as directional]**

### 2.6 Metric

**Macro-averaged ROC-AUC** over the 12 label columns. **[C]**

Three consequences that should shape every design decision:

1. **AUC is rank-only.** Absolute calibration is worth nothing. A prediction vector's *ordering*
   within each label column is the entire score. This is why soft/graded targets beat binary ones
   (§8 bet A) — a binary target has 2 distinct values and cannot rank within either block.
2. **Macro means rare labels cost the same as common ones.** A label with 10 % prevalence is worth
   exactly as much as one at 60 %. A single shared loss quietly under-serves the rare ones; the
   metric does not. Budget per-label capacity accordingly.
3. **Per-label AUC is computed independently.** Cross-label calibration does not matter either. You
   may (and should) tune each label's head, resolution and slice sampling separately.

Submission is a CSV with `StudyInstanceUID` plus the 12 label columns as continuous probabilities.
A valid `sample_submission.csv` sets every label to 0.5. **[S]** **Do not threshold to binary** —
that destroys rank information and is a self-inflicted wound under AUC.

### 2.7 Format, constraints, timeline, prizes

| | |
|---|---|
| Type | **Research Code Competition** — submit a notebook **[C]** |
| Runtime | **≤ 9 hours**, **internet off** at submission **[C]** |
| Test set | ~1,300 studies; public LB ≈ 30 %, private ≈ 70 % **[S]** |
| Opened | 2026-07-30 **[S]** |
| **Entry / team-merger deadline** | **2026-10-15** **[C]** |
| **Final submission** | **2026-10-22** **[C]** |
| Winners announced | November 2026 **[C]** |
| Winners' obligations | Training code, model weights, method description, video — reportedly due ~5 Nov **[S]** |
| Prize pool | **$77,000**, including a **first-ever award for the most efficient models** **[C]** |
| Efficiency Track | reportedly **$18,000 across 3 places** **[S]** — see §9 |

**Public LB is ~390 studies.** That is small enough that chasing it is a trap; the private 70 % is
what pays. §10.

### 2.8 Rules constraints to respect

- **Report text must not go to a hosted LLM API** pending a host ruling — see §1.6. **[S/?]**
- **External datasets** (MRNet, OAI, fastMRI+, SKM-TEA) — whether click-through-agreement datasets
  count as "freely and publicly available" is **unresolved**. Treat as **blocked until the host
  rules**, not merely deprioritised. **[S/?]** §7.4 catalogues them for when it clears.
- Public sharing rules apply as normal for a Kaggle competition — anything shared must be shared
  with everyone on the forum.

---

## 3. The data

> **Provenance warning.** Nothing in this section was measured by us — see §1.4 and §1.7. It comes
> from two independent public repositories by teams working this competition. §3.9 shows where they
> corroborate each other. §11 is the checklist that converts these to [M].

### 3.1 Scale

| Fact | Value | Conf |
|---|---|---|
| Total dataset size | **≈ 569.76 GB** | [S] |
| Total files | **819,640** (one source) / **819,078** train DICOMs (other source) | [C] |
| **Train studies** | **4,407** | **[C] — both repos independently** |
| **Train series** | **24,371** | **[C] — both repos independently** |
| Series per study | median **5**, range 3–14 | [S] |
| Slices per series | 20–45 typical, median ~30, p99 = 160, max = 320 | [S] |
| Test studies | ≈ **1,300** | [S] |
| Reports present | 4,407 / 4,407 (**100 %**) | [S] |

The 562-file difference between the two counts is most likely train-only vs train+test bookkeeping.
Not load-bearing.

### 3.2 The single most important number: **58**

| | | Conf |
|---|---|---|
| Studies with **all 12 official labels** | **58 (1.3 %)** | **[C] — both repos** |
| Studies with a **report but no labels** | **4,349 (98.7 %)** | **[C] — both repos** |

`train.csv` has 4,407 rows. Only 58 carry the 12 labels. **There is no meaningful supervised image
dataset in this competition — you have to manufacture one.** This is the fact that reframes
everything; see §4.

Two further properties of the 58 that matter enormously for how you use them: **every one of the 58
has at least one positive finding**, and prevalence in the 58 appears roughly **2× enriched**
relative to the corpus. **[S]** They are **not a random sample**. Consequences:

- The 58 are a **direction check, not a scoreboard.** At n=58 the standard error on an AUC is
  roughly ±0.06; you cannot resolve 0.01-level differences and you must not tune on them.
- If 9–11 of 12 labels move the same way, that is signal. One label improving is noise.
- Hold them out of training entirely, or the only honest evaluation you own is gone.

### 3.3 Reports — multilingual, and the language is a site fingerprint

Detected over all 4,407 reports **[S]**:

| Language | n | % |
|---|---|---|
| English | 1,736 | 39.4 |
| Turkish | 546 | 12.4 |
| Spanish | 532 | 12.1 |
| Latin-other (Croatian/Serbian, Flemish, …) | 487 | 11.1 |
| Greek | 321 | 7.3 |
| German | 257 | 5.8 |
| Cyrillic (Bulgarian/Russian) | 220 | 5.0 |
| French | 159 | 3.6 |
| Dutch | 148 | 3.4 |

RSNA's own announcement says **12 languages, 16 sites, five continents** **[C]** — so the 9 buckets
above are a detector's coarse grouping, not the true count. Expect at least 12.

Other measured report properties **[S]**:

- Median report **≈ 129 words**.
- De-identified with `[DATE]` / `[TIME]` placeholders.
- **Text-substitution artifacts**: numeric fragments replaced by the token `intact`, producing
  strings like `intact9xintact4cm`. **Warning against naive numeric parsing** — and a cleaning pass
  is warranted.
- **Section headers range from 1.6 % (German) to 99.8 % (Spanish)** of reports.
- **Reports under 50 words range from 0.6 % (French/Greek) to 33.9 % (Latin-other)** and 28.8 %
  (Spanish).

That last pair is a **hazard, not a curiosity**. A 30-word report does not enumerate negatives.
Mapping "not mentioned" → 0 is right for a long structured report and wrong for a short one — and
because report format tracks language, which tracks site, **this injects a site-correlated bias
directly into your training labels.** Handle it: confident 0 in long structured reports; soft and
down-weighted in short ones. See §8 bet A.

### 3.4 Series structure — 6 slots, not 12

`Fluid_Sensitive` and `Fat_Suppression` are **perfectly correlated** — the cross-tab is exactly
diagonal (10,361 series at (0,0), 14,010 at (1,1), **zero** off-diagonal). **They are one bit, not
two.** **[S]** So the series space is **3 planes × 2 sequence types = 6 slots**:

| Slot | series | % of studies with ≥1 | mean slices | p50 |
|---|---|---|---|---|
| Axial \| fluid-sensitive | 4,719 | **100.0 %** | 39.5 | 32 |
| Sagittal \| structural | 5,197 | 96.8 % | 33.7 | 30 |
| Coronal \| fluid-sensitive | 4,624 | 96.4 % | 28.5 | 30 |
| Sagittal \| fluid-sensitive | 4,667 | 94.2 % | 35.5 | 29 |
| Coronal \| structural | 3,985 | 77.3 % | 28.0 | 30 |
| Axial \| structural | 1,179 | **19.4 %** | 41.0 | 32 |

Four slots cover >94 % of studies; the sixth is present for 1 study in 5 and is near-useless.
**A presence mask over slots is required regardless** — the hidden test set will have gaps too.

`train_series.csv` columns **[S]**: `StudyInstanceUID`, `SeriesInstanceUID` (matches the folder
name), `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` ∈ {Sagittal, Coronal, Axial}.

DICOM layout **[C]**:
```
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm   # one slice per file
test_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
train.csv  train_series.csv  test.csv  test_series.csv  sample_submission.csv
```

### 3.5 DICOM headers are richer than the CSVs — and available at test time

**86 tags survive de-identification.** **[S]** The ones that matter:

| Tag | Why it matters |
|---|---|
| `SeriesDescription` | Free text, e.g. `pd_tse_tra_d`. Richer than the 3 CSV columns |
| `Laterality` | Explicit `L`/`R` — but see §3.7, it is missing half the time |
| `PatientSex` | Present in DICOM though **absent from `train.csv`** despite the data description claiming otherwise (documented doc bug) |
| `PatientID` | Pseudonymous; lets you check for repeat patients |
| `Manufacturer`, `ManufacturerModelName`, `SoftwareVersions`, `MagneticFieldStrength`, `ReceiveCoilName`, `ImagingFrequency` | **Scanner fingerprint** = site proxy |
| `TR`, `TE`, `TI`, `FlipAngle`, `EchoTrainLength`, `PixelBandwidth`, `SliceThickness`, `SpacingBetweenSlices`, `PixelSpacing`, `AcquisitionMatrix` | Full MR physics |
| `ImagePositionPatient`, `ImageOrientationPatient` | **True 3-D slice ordering and geometry** |

**`SeriesDescription` is useful but not a replacement** **[S]**: 82.3 % usable, 11.8 % is the literal
placeholder `DummySeriesDesc!`, 5.9 % null; **558 studies (12.7 %) have the placeholder on every
series**. Where parseable, plane agrees with `Anatomical_Plane` **99.7 %** of the time — but only
74.4 % is parseable. And the placeholder rate is vendor-specific (one Siemens spelling 100 %,
Philips 29.2 %, GE 19.4 %), which makes **its presence a site marker, not a description**.
→ **Decision: CSV columns primary, `SeriesDescription` auxiliary only.**

### 3.6 Physical scale — the correctness requirement most teams will miss

Measured distributions **[S]**:

| Field | p1 | p25 | p50 | p75 | p99 |
|---|---|---|---|---|---|
| `PixelSpacing` (mm) | 0.137 | 0.250 | 0.312 | 0.391 | 0.703 |
| Rows | 256 | 384 | 512 | 640 | 1024 |
| `SliceThickness` (mm) | 0.6 | 3.0 | 3.0 | 3.5 | 4.5 |
| `SpacingBetweenSlices` (mm) | 1.0 | 3.3 | 3.5 | 4.10 | 6.20 |
| **Field of view (mm)** | **130** | 160 | 160 | 170 | 205 |

**`PixelSpacing` spreads by 5.14×.** A fixed-pixel resize therefore feeds the network anatomy whose
physical scale differs several-fold between studies. **Crop to a constant millimetre extent, then
resize.** This is not a refinement, it is a correctness requirement.

`CROP_MM = 130` is the right constant, and two teams reached it independently: **≥130 mm covers
99.57 % of series; ≥140 mm covers only 94.91 %** — 130 sits exactly at the knee of the curve. **[C]**

Resulting pixel pitch, and the Nyquist argument:

| Target px | mm/px at 130 mm crop | Resolves a ~1 mm meniscal tear? |
|---|---|---|
| 224 | 0.580 | **✗** — Nyquist needs ≤0.5 mm |
| **336** | **0.387** | ✓ |
| 448 | 0.290 | ✓, at ~1.8× the cost of 336 |

Median native spacing is 0.312 mm, so 448 px is roughly native and 336 px is a mild downsample.
**Cache at 336 px.** Downsampling later is free; upsampling is impossible. This prediction was
tested — see §5.4, where the two labels that failed below chance were exactly the fine-detail ones
at 224 px.

### 3.7 Laterality — half the dataset does not say which knee it is

| | | Conf |
|---|---|---|
| Series with **no** `Laterality` tag | **12,367 / 24,371 (50.7 %)** | [S] |
| Studies with the tag on ≥1 series | 2,204 (50.0 %) | [S] |
| **Studies with no laterality anywhere** | **2,203 (50.0 %)** | [S] |
| **Bilateral studies** (≥2 distinct sides) | **25 (0.57 %)** | [S] |

Missingness is **site-linked**: German/Dutch/French reports carry the tag ~100 % of the time;
English 38 %, Turkish 35 %, Greek 37 %. And **`GE MEDICAL SYSTEMS` (4,914 series) never carries it
at all.**

**Geometry recovers the side.** Compute the image-centre x in patient coordinates —
`IPP[0] + ½·cols·ps·IOP[0] + ½·rows·ps·IOP[3]`; DICOM LPS so **+x = patient Left** — and validate
against the ~12,000 series that *do* carry the tag:

| Threshold | Coverage | Sign-rule accuracy |
|---|---|---|
| \|x\| ≥ 0 mm | 100 % | 97.4 % |
| \|x\| ≥ 10 mm | 98.4 % | 98.3 % |
| **\|x\| ≥ 20 mm** | **97.3 %** | **98.5 %** |
| \|x\| ≥ 40 mm | 93.0 % | 98.7 % |

Left knees centre at **+83 mm**, right at **−79 mm** — clean separation.

A third, independent source exists for training studies: **reports state the side in their first
line** (`SOL DİZ`, `MR Knie Rechts`, `MRI ΑΡΙΣΤΕΡΟΥ ΓΟΝΑΤΟΣ`). A report-side extractor was measured
at **98.8 % accurate where it fires**, covering 37.2 % of studies — enough to audit geometry on the
vendors where the tag is absent. Cross-validated agreement of geometry vs report-derived side:

| Vendor | n | Agreement |
|---|---|---|
| SIEMENS | 818 | 98.9 % |
| PHILIPS | 184 | 96.2 % |
| **GE** | **485** | **92.6 %** |
| TOSHIBA | 126 | 87.3 % |
| FUJIFILM | 16 | 81.2 % |
| HITACHI | 8 | 37.5 % (n=8 — ignore) |
| **Overall** | **1,638** | **95.4 %** |

→ **Shipped rule: DICOM tag when present, else geometry at \|x\| ≥ 20 mm. Coverage 100 %, expected
accuracy ~97–98 %.** For *training* you can do better: tag → report → geometry in that order, and
down-weight studies where geometry and report disagree.

**Bilateral studies are 25 (0.57 %) — a footnote, not a workstream.** The host stated that each
bilateral study was individually reviewed and the report text or DICOM metadata adjusted so
participants can disambiguate. Labels describe **one** knee. **[S]**

### 3.8 Decode cost — much cheaper than feared

**Every training series is `Explicit VR Little Endian` (uncompressed). 100 %.** **[S]** The data
description lists four transfer syntaxes including JPEG 2000; in the training data there is exactly
one, and it is the fastest. Measured decode: **5.2 ms/slice**.

| Task | Slices | 1 thread | ~4 processes |
|---|---|---|---|
| Full train cache (6 slots × 16) | ~423,000 | ~37 min | **~10–15 min** |
| Test inference (1,300 studies) | ~125,000 | ~11 min | **~3–4 min** |

Two consequences. The cache build is a **15-minute CPU job**, not an overnight one. And **decoding
is nearly free at inference** — which is decisive for the Efficiency Track (§9), because the runtime
budget can go almost entirely to the model.

⚠️ **But the hidden test set may contain the compressed syntaxes the description mentions.** The
inference notebook must have `pylibjpeg` / `gdcm` available and handle them, or it will crash on
data it has never seen. This is a cheap insurance policy against a catastrophic failure mode.

### 3.9 Corroboration — where the two independent sources agree

This is the reason §3 is usable at all rather than one team's word.

| Fact | Repo A (homeshwarnelakurthi) | Repo B (JunhaoLiXD) | Agree? |
|---|---|---|---|
| Train studies | 4,407 | 4,407 | ✅ |
| Train series | 24,371 | 24,371 | ✅ |
| DICOM files | 819,640 | 819,078 | ~ (train vs all) |
| Gold-labelled studies | 58 | 58 | ✅ |
| Report-only studies | 4,349 | 4,349 | ✅ |
| The 12 label names | identical list | identical list | ✅ |
| Median slices/series | ~30 | 32 | ✅ |
| Metric | macro ROC-AUC | column-averaged AUC | ✅ |
| Directory layout | as §3.4 | as §3.4 | ✅ |

Different codebases, different label schemes, different conclusions — same numbers. The counts in
§3.1–3.2 should be treated as near-certain. The *interpretive* claims (severity thresholds, site
leakage magnitudes, extractor scores) rest on a single source and are marked [S] individually.

---

## 4. What kind of problem this actually is

Four framings, in descending order of how much they should change what you build.

### 4.1 It is a weak-supervision problem wearing a computer-vision costume

58 labelled studies cannot train a vision model. 4,349 reports can — *if* you convert them into
targets. **Whoever converts multilingual free-text reports into the best training targets sets the
ceiling for every vision model trained downstream.** That is the competition. The CNN is
downstream plumbing.

This is why the public LB has a dense cluster at exactly 0.891 (§5.2): everyone forked the same
baseline's label extractor, so everyone inherits the same ceiling.

### 4.2 The labels and the reports are two different measuring instruments

Ground truth: **two MSK radiologists reading images, adjudicated by a third, under explicitly
severity-thresholded criteria, with "on the fence" graded negative** (§2.4, §2.5).
Reports: **one clinical radiologist, routine care, writing prose for a referring physician.**

Report-derived labels reportedly agree with ground truth only **~82.5 %** of the time. **That 18 %
gap is not noise — it is systematic, monotone, and therefore correctable.** Nearly all of it is
reports mentioning findings the rubric grades negative because they are mild, borderline, chronic,
or non-surface-reaching. This is the single largest exploitable asymmetry in the competition and it
is the basis of bet A (§8).

### 4.3 AUC being rank-only turns that asymmetry into a design

Because the metric is rank-based, you do not want a binary target at all. You want a target that
**preserves the severity ordering**:

```
normal cartilage  <  mild chondropathy  <  grade 3 focal  <  grade 4 over 2 cm
"no effusion"     <  "trace"            <  "small"        <  "moderate"  <  "large/massive"
intact ACL        <  "mild signal change" < "partial tear" < "complete rupture"
```

A model trained on a severity continuum ranks correctly **under any threshold**, which immunises it
against the exact report-vs-image threshold mismatch costing everyone else ~18 % label error. A
binary target has 2 distinct values and **cannot rank within either block** — every positive ties
with every other positive. Granularity *is* the gain.

**This was tested.** On the 58 gold studies, a presence extractor scored macro-AUC **0.7443**; the
same code with severity-graded outputs scored **0.7907** — **+0.0464, improving 11 of 12 labels**,
bootstrap 95 % CI **[+0.0081, +0.0818]**, P(Δ>0) = 99.1 %. **[S]** Split by rubric family:

| Rubric family | Presence | Severity | Δ | Improved |
|---|---|---|---|---|
| **Magnitude** (OA ×3, Effusion, Synovitis, Baker's) | 0.7198 | 0.7845 | **+0.0647** | **6/6** |
| **Categorical** (ACL, MCL, menisci ×2, Contusion, Fracture) | 0.7688 | 0.7970 | +0.0282 | 5/6 |

**Caveat, stated by the source itself:** three iterations against the same 58 studies, so +0.0464 is
partly in-sample and the true out-of-sample gain is lower. **Treat the direction as established and
the magnitude as an upper bound.**

### 4.4 The rubric splits the 12 labels into two kinds of question

Per-label clause-level co-occurrence of severity language **[S]**:

| Rubric asks | Labels | Severity word in same clause |
|---|---|---|
| **"how much?"** | Synovitis 56.5 %, PF OA 46.6 %, Effusion 46.4 %, Medial OA 37.9 %, Baker's 29.9 %, Lateral OA 28.2 % | 28–57 % |
| **"what kind?"** | Contusion 26.8 %, MCL 23.3 %, Medial Meniscus 22.8 %, ACL 20.5 %, Fracture 16.0 %, Lateral Meniscus 15.4 % | 15–27 % |

Magnitude words answer a magnitude rubric. But a meniscal tear is graded by **whether signal reaches
the surface**, an ACL tear by **complete vs partial vs degeneration** — those are categories, not
degrees. → **Build two extractors, not one:** magnitude vocabulary for the first six, categorical
vocabulary for the second six. 82.1 % of reports carry severity or grade language *somewhere*, so
the raw material is there.

### 4.5 Site leakage is large, measured, and will lie to you

Three independent measurements, all pointing the same way **[S]**:

| Probe | Random folds | Site-grouped folds | Gap |
|---|---|---|---|
| DICOM header metadata only, no pixels | **0.6515** | **0.5981** | **+0.053** |
| Series composition alone (4 CSV columns) | 0.5954 | — | — |
| A trained resnet34 vision model, epoch 12 | **0.8412** | **0.7049** | **+0.1363** |

Two conclusions. First, **there is no metadata shortcut** — 0.598 grouped is barely above chance, so
the leaderboard reflects genuine image reading. Second, and far more useful: **random K-fold
overstates performance by 0.05 from headers alone and by 0.14 once a CNN sees pixels.** The model is
reading **scanner signature out of the pixels themselves** — noise texture, reconstruction kernel,
native resolution — at 2.6× the rate the headers alone explain.

Watch it grow over training **[S]**:

| Epoch | Grouped | Random | Gap |
|---|---|---|---|
| 1 | 0.516 | 0.562 | +0.046 |
| 4 | 0.671 | 0.741 | +0.070 |
| 8 | 0.702 | 0.819 | +0.117 |
| 12 | 0.705 | **0.841** | **+0.136** |

Random validation climbs steadily while grouped **plateaus at ~0.70 from epoch 6**. Everything after
epoch ~7 is the model learning *which scanner took the picture*. A team validating on random folds
reads 0.841, feels good, and finds out on the private leaderboard.

**But do not treat grouped CV as truth either.** Train and test are drawn from **the same 16 sites**,
so the test set is not an unseen-site holdout — grouped CV is *pessimistic*, random is *optimistic*,
neither is the target. **Report both, every run.** The gap between them is a live diagnostic of how
much the model leans on site rather than anatomy, and driving it down is itself an objective.

**Fold scheme.** The 5-part scanner fingerprint fails: it makes 3,262 groups of which 2,668 are
singletons, because `ImagingFrequency` varies per *scan* (63.685238 vs 63.685256), not per scanner.
Candidates over 4,407 studies **[S]**:

| Scheme | Groups | Largest | Singletons | Groups ≥50 |
|---|---|---|---|---|
| manufacturer | 7 | 44.3 % | 0 | 4 |
| manu \| model | 46 | 16.8 % | 8 | 21 |
| language | 10 | 39.4 % | 1 | 9 |
| **language \| manu \| model** | **75** | **5.8 %** | 15 | **27** |

**Language is an excellent site proxy** — Cyrillic is 100 % Philips; Dutch, German and Greek are
100 % Siemens. Those are single institutions showing through.
→ **Group folds on `language | manufacturer | model`.** 75 groups, none above 5.8 %, 27 with ≥50
studies. It is the closest thing to an institution key that can be built from the released data.

**Patient grouping is not needed:** 4,407 distinct `PatientID`s for 4,407 studies, **zero repeats**.
**[S]** (Caveat: IDs may have been re-randomised per study during de-identification, which would
hide real repeats — but site grouping partially covers that risk anyway.)

---

## 5. The public landscape

### 5.1 Leaderboard shape (as of ~2026-08-08, competition day 9) **[S]**

| Rank | Score |
|---|---|
| 1 | **0.939** |
| 2 | 0.933 |
| 3 | 0.929 |
| … | dense cluster pinned at exactly **0.891** |

676 teams at that date. RSNA's prior challenges drew 1,147 (2025 aneurysm) and 1,874 (2024 lumbar)
teams **[C]**, so expect the field to be several times larger by the 22 Oct deadline.

**A dense cluster at one identical score is the unmistakable signature of a public baseline being
forked wholesale.** 0.891 is table stakes, not an achievement. The interesting question is what the
0.939 team does that the 0.891 fork does not — and the gap is only 0.048, which at this stage of a
competition usually means the top teams have not yet deployed their real solutions.

⚠️ **This snapshot is ~5 weeks stale.** By 2026-09-11 the whole distribution has almost certainly
moved up. **Re-read the leaderboard before setting any target.** §11.

### 5.2 The public baseline — `pilkwang/rsna-knee-baseline-v1` **[S]**

221 votes. Reported public score **0.809** for the baseline itself; the 0.891 cluster is the
community's improved forks of it. **It is not a toy.** It already does:

- a hand-built **multilingual rule extractor** covering 9+ languages, with negation, normality and
  uncertainty lexicons, clause splitting and heading attachment;
- **physical-scale normalisation** — `CROP_MM = 130`, then resize to 224 or 336;
- the **6 plane × sequence slots** with a presence mask;
- **DINOv2-small**, last 6 blocks unfrozen, `lr_backbone = 8e-6`, `lr_head = 1e-3`, 10 epochs;
- **no horizontal flip augmentation** — correct, because flipping swaps medial and lateral, which
  are distinct labels in 5 of 12 targets;
- an 8-hour internal budget to fit inside the 9-hour cap.

**The floor is high and well-defended.** Every "obvious" preprocessing idea is already in the public
baseline. Differentiation has to come from the label side (§8 A), the fold discipline (§8 C), the
resolution/sampling side (§8 D/E) or the efficiency side (§9) — not from re-discovering `CROP_MM`.

### 5.3 Other public artifacts seen **[S]**

| Notebook / asset | Note |
|---|---|
| `pilkwang/rsna-knee-baseline-v1` | The reference baseline, 0.809 → forks at 0.891 |
| `debugendless/rsna-knee-abnormality-detection-baseline` | Another public baseline |
| `romantamrazov/rsna-knee-dinosaur-v2` | DINOv2 variant |
| `xiaoleilian/rsna26-knee-eda` | Public EDA |
| `mpwolke/rsna-knee-abnormality-detection-2026-dcm` | DICOM handling notebook |
| **`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`** | **The Efficiency leaderboard notebook** — Kaggle-official. Read it to confirm §9's formula |
| `riachk/rsna-knee-abnormality-detection-tool` | A published Kaggle Model |

### 5.4 A real training result from a competing team — the most useful data point available **[S]**

One team's first end-to-end run, logged with full detail. This is worth more than any writeup
because it is a measured baseline with an honest post-mortem.

**Setup:** resnet34, slices-as-channels (12 ch), masked attention over the 6 slots; cache 224 px /
12 slices / `CROP_MM` 130, canonicalised to one side; severity soft labels with `confidence` as a
per-sample loss weight; GroupKFold on `language|manufacturer|model`, fold 0 of 5; 3,479 train /
870 val, **58 gold excluded from training entirely**; 12 epochs, OneCycle, AdamW, no h-flip;
**1× Tesla T4, runtime 1,224 s (20.4 min)**.

**Results:**

| Metric | Value |
|---|---|
| Grouped val AUC (vs weak labels) | 0.7049 |
| Random val AUC (vs weak labels) | 0.8412 |
| **Site-reliance gap** | **+0.1363** |
| **Gold macro AUC (58 held-out, true labels)** | **0.6739** (best, epoch 9) |

Per-label AUC on the 58 gold studies:

| Label | n_pos | AUC | |
|---|---|---|---|
| Effusion | 35 | **0.922** | the proof of concept |
| Contusion | 19 | 0.776 | |
| Baker's | 12 | 0.748 | |
| ACL | 24 | 0.719 | |
| Lateral Meniscus | 23 | 0.704 | |
| Fracture | 18 | 0.672 | |
| Synovitis | 27 | 0.662 | |
| Lateral OA | 11 | 0.658 | |
| Medial OA | 15 | 0.650 | |
| PF OA | 21 | 0.574 | |
| **Medial Meniscus** | 26 | **0.476** | **below chance** |
| **MCL** | 9 | **0.420** | **below chance** |

**Four things this tells us, and they are the most actionable facts in this document:**

**(a) Vision capacity is currently the bottleneck, not label quality.**

| Predictor of the 58 gold labels | Macro AUC |
|---|---|
| Report labeler v3, from **text** | **0.791** |
| The trained model, from **images** | 0.674 |

**The text labeler beats the vision model by 0.117.** The targets already carry more signal than
the model extracts from pixels. So effort belongs on the model — resolution, backbone, slice
sampling — before further squeezing of the labeler. *(Caveat both ways: 0.791 is partly in-sample
over three iterations; 0.674 is one fold of resnet34 at 224 px. The direction is what matters, and
it is not close.)*

**(b) The two failing labels were predicted in advance, by the Nyquist argument of §3.6.**
`Medial Meniscus` 0.476 and `MCL` 0.420 are below chance. Both are fine-detail findings, and at
`CROP_MM` 130 / 224 px the pitch is 0.58 mm against a ~1 mm tear. **336 px gives 0.387 mm and clears
Nyquist.** Two separable hypotheses, and the experiment distinguishes them cleanly:
- *Resolution* — rebuild the cache at 336 px. Predicts menisci improve and Effusion (already 0.922,
  a large finding) does not.
- *Slice sampling* — on sagittal the menisci sit near the **ends** of the stack, and the sampling
  band was (0.18, 0.82) with only 12 slices. They may simply be sampling past them. Test by widening
  the band and raising sagittal slice count.

**(c) Do not tune on MCL.** 9 positives. The text labeler also failed it (−0.075). Anything that
"fixes" a label with 9 positives is fitting noise.

**(d) Effusion at 0.922 proves the pipeline.** A well-posed, visible finding reaches 0.922 through
this pipeline in 20 minutes on a resnet34. Geometry, canonicalisation, slot masking and the label
join all work. **The weak labels are usable.** Nothing structural is broken.

### 5.5 The other competitor's version history — what moved the LB **[S]**

| Version | Change | Kaggle public score |
|---|---|---|
| V01 | rule-weak 3-plane 2.5D EfficientNet-B0 | **0.613** |
| V02 | fold-safe calibrated soft labels, gold weight 8 | OOF **collapsed to base rate** (pred std ~0.05) |
| V03 | hierarchical **state-specific** soft-label priors, ordering + margin constraints, 15 % zero-support confidence floor, prediction-spread collapse diagnostic | **0.664** |
| V04 | **DINOv2-small (ViT-S/14)** @224, **laterality normalisation**, **target-specific attention pooling** | implemented, untrained |

Two lessons worth more than the scores:

1. **V02's collapse is a real failure mode.** Miscalibrated soft targets can drive a model to predict
   the base rate for every study — OOF prediction std ~0.05, AUC ≈ 0.5. **Instrument for it:**
   log per-label prediction standard deviation every epoch and fail loudly if it collapses. This
   costs nothing and would have saved that team a full training run.
2. **The fix that recovered it** was making the uncertainty live in the right variable. Their
   `UNMENTIONED_PRIOR` had been set to P(positive | not mentioned) — 0.18 for Synovitis. Under AUC
   the absolute value carries **no information at all**; only rank does. A prior of 0.18 placed every
   silent study **above** a report saying "mild synovitis" at 0.15 — **inverting the evidence**,
   since a mention is evidence *for* a finding. Moving the rank estimate into `severity` and the
   uncertainty into `confidence` took Synovitis from 0.518 → 0.676. **Severity carries rank;
   confidence carries uncertainty. Never conflate them.**

### 5.6 Two known bugs, documented, so we do not repeat them **[S]**

- **Negation ordering.** A categorical scorer that tests negation **last** is broken: `"medial
  meniscus: no tear"` matches `TEAR` and scores positive, because the negation branch is unreachable
  whenever any pathology word appears — which is nearly always. Their magnitude scorer tested
  negation first and their categorical scorer did not; **that single ordering difference** was the
  entire reason one family gained +0.049 and the other +0.000. Medial Meniscus went −0.020 → +0.067
  on the fix. **Test negation first. Always.**
- **Blank strings are missing values.** Counting only `NaN` for `Laterality` gave 20.9 % missing; the
  true figure including blank strings is **50.7 %**. The same error inflated the bilateral-study
  count from 25 to 83. **Treat `""` as null everywhere.**

---

## 6. Prior competitions surveyed

### 6.1 How the survey was scoped

You asked me to work through five Kaggle tag pages — Medicine (12028), Binary Classification
(14201), Multilabel Classification (16636), Image Classification (16686), Computer Vision (13207) —
and pick the competitions whose solutions transfer. **Those pages are on `kaggle.com` and were
unreachable (§1.2)**, so the enumeration was done from search plus prior knowledge of the Kaggle
medical-imaging catalogue, and then **verified by going to the winners' GitHub repositories**, which
were reachable and which contain the actual pipelines.

Selection criterion, applied deliberately narrowly: **a prior competition earns a place here only if
it shares a structural feature with this one** — 3-D/multi-series medical volumes reduced to a
study-level multilabel prediction, an AUC-family metric, a code-competition runtime cap, weak or
noisy labels, or multi-site domain shift. Competitions that are merely "medical" or merely
"multilabel" were dropped; there is no value in cataloguing them.

### 6.2 The selected set

| Competition | Year | Why it transfers | Evidence obtained |
|---|---|---|---|
| **RSNA 2024 Lumbar Spine Degenerative Classification** | 2024 | **Closest relative.** Multi-plane **MRI**, multiple named anatomical levels, per-condition severity grading, side-specific labels | Writeup summary + repos |
| **RSNA 2023 Abdominal Trauma Detection** | 2023 | Study-level multilabel from a 3-D volume; the canonical 2.5D **CNN + RNN** design | **1st and 2nd place repos cloned** |
| **RSNA 2022 Cervical Spine Fracture Detection** | 2022 | Per-level labels from a volume; localise-then-classify | Writeup summary |
| **RSNA 2025 Intracranial Aneurysm Detection** | 2025 | Most recent RSNA; 13 anatomical locations + multi-modality; ROI-crop framing | **7th place (MIC-DKFZ) repo cloned** |
| **RSNA Screening Mammography Breast Cancer Detection** | 2023 | Extreme class imbalance, ROI detection first, and **hard inference-runtime engineering** | **1st place repo cloned** |
| **HMS Harmful Brain Activity Classification** | 2024 | **Soft/probabilistic labels from disagreeing annotators** — structurally our label problem | Writeup summary |
| **SIIM-ISIC Melanoma / VinBigData CXR** | 2020/21 | Multi-site domain shift under AUC; per-site distribution differences | Summary only |

### 6.3 The recurring winning pattern across every RSNA volume competition

Five competitions, five different organs, **one architecture family**. This is the strongest
prior-art signal available and it should be the default design:

```
Stage 1  — LOCALISE.    Segmentation / detection / keypoint model finds the anatomy,
                        producing a crop or a slice-range. Cheap model, small data.
Stage 2  — CLASSIFY.    2.5D: adjacent slices as channels into a 2D ImageNet-pretrained
                        backbone. Never a from-scratch 3D CNN.
Stage 3  — AGGREGATE.   A sequence head (GRU / LSTM / transformer / attention pooling)
                        fuses per-slice features into the study-level prediction.
Stage 4  — ENSEMBLE.    Several backbones × seeds × folds; weight per target, not globally.
```

Concrete instances:

| Competition | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| RSNA 2023 abdominal, 1st | 3-D segmentation → organ crop | CoaT-Lite / EffNetV2-S @384, 96 slices → (32,3,H,W) | GRU over 32 steps |
| RSNA 2023 abdominal, 2nd | EffNetV2-rw-t organ-presence + 3-D resnet18 crop | MaxViT-tiny@512 / ConvNeXtV2-tiny, 3 adjacent frames as channels | Heavily tweaked bidirectional LSTM, **organ-conditioned pooling**, **independent per-organ logits** |
| RSNA 2022 cervical, 1st | vertebra segmentation | EffNetV2-S / ConvNeXt, 15 slices + neighbours + masks as channels | LSTM per vertebra |
| RSNA 2024 lumbar, 1st | CenterNet keypoints, EfficientNet-B6 + FPN → per-disc-level crops | separate "center" and "side" classifiers | per-level heads |
| RSNA 2023 mammography, 1st | YOLOX-nano 416 breast-ROI detector | ConvNeXt-small on ROI crops, 4 folds | — (single image) |
| RSNA 2025 aneurysm, 7th | 200×160×160 mm ROI crop | nnU-Net ResEnc-M 3-D, multi-task seg+cls | — |

### 6.4 What each one specifically teaches *this* competition

**RSNA 2024 Lumbar Spine — the closest relative.** **[S]**
The winning team ("Avengers", also awarded the Educational Merit Award) built a **two-stage**
system: a CenterNet-style 2-D keypoint detector (EfficientNet-B6 backbone, FPN neck) locating each
disc level on sagittal images, then converting image coordinates to **real-world coordinates** to
assign disc levels to axial slices; then separate classifiers for centre findings and side findings.
Two details worth stealing outright:
- They **generated pseudo-labels with the trained model on unused data**, then used all of it.
- They **manually reviewed every annotation and hand-corrected errors**, having recognised label
  noise existed. In a competition where 98.7 % of labels are machine-derived, a few hours of human
  eyeballing on the extractor's worst disagreements is very likely the highest-yield hour available.

→ **Transfer:** the "centre finding vs side finding" split maps almost exactly onto our
**side-specific (5) vs central (7)** label split, and argues for separate heads rather than one.

**RSNA 2023 Abdominal Trauma — the canonical 2.5D recipe.** **[C — 1st and 2nd repos both read]**
1st place: 96 equidistant slices per study → reshaped to `(32, 3, H, W)` with 3 **adjacent slices as
channels**, 384×384, CoaT-Lite-Medium and EfficientNetV2-S backbones, GRU head, **4-fold GroupKFold
at patient level**. Targets were built by multiplying the study-level label by **organ visibility**
derived from segmentation masks — i.e. *a slice only carries the label to the extent the relevant
anatomy is actually in it.*
2nd place added: **organ-conditioned pooling** (pool RNN features using the segmentation
probabilities) and **independent per-organ logits that see only their own organ's features**. Also
notable: **heavy augmentation + CutMix at p=0.5–1.0**, Ranger optimiser, and that `maxvit_tiny_tf_512`
beat everything else.

→ **Transfer, and this is the big one:** *"a slice only carries the label to the extent the relevant
anatomy is in it"* is precisely our per-label slot problem. PF OA is visible axially; the cruciates
sagittally; the collaterals and compartments coronally. **Weight each label's pooling by which
slot/slice can actually see it** — see §8 bet E.

**RSNA 2022 Cervical Spine — per-level classification.** **[S]**
15 slices per vertebra, neighbours **and segmentation masks** concatenated as extra channels,
EfficientNetV2-S/ConvNeXt + LSTM. → **Transfer:** feeding a mask or an anatomical prior as an extra
input channel is cheap and repeatedly wins.

**RSNA 2025 Aneurysm — the most recent RSNA, and a cautionary note.** **[M — repo read directly]**
The MIC-DKFZ (nnU-Net) team placed 7th/1147 with a **3-D nnU-Net ResEnc-M**, batch 32, multi-task,
on a **200×160×160 mm ROI crop** — note the **millimetre-defined crop**, the same principle as our
`CROP_MM = 130`. But read their hardware line: **4× A100 40 GB for 4.5 days, ~130 s/epoch.**
→ **Transfer:** the mm-crop principle, yes. The architecture, **no** — it is unreachable inside a
9-hour Kaggle T4 budget. This is the clearest evidence that **2.5D, not 3-D, is the correct choice
for our constraints.** The 1st place in that competition was titled *"Location-Aware Aneurysm
Detection via Vessel-ROI Masking"* — again, localise first.

**RSNA 2023 Mammography — the efficiency masterclass.** **[M — repo read directly]**
1st place ran **YOLOX-nano @416 for ROI detection → ConvNeXt-small classification on the crop**, and
**converted both models to TensorRT** for inference. A 4-fold split; the ROI detector was trained on
just **521 images**.
→ **Transfer:** (a) a tiny detector trained on a few hundred hand-annotated images is enough to
localise anatomy, so ROI-cropping is cheap to add; (b) **inference-engine optimisation is a
first-class lever**, which matters directly for §9. On Kaggle, ONNX Runtime or `torch.compile` +
fp16 are the accessible versions.

**HMS Harmful Brain Activity — our label problem in another costume.** **[S]**
Labels were **vote distributions from disagreeing annotators**, scored by KL divergence. The
dominant winning pattern was **two-stage training: pretrain on the large noisy-label pool, then
fine-tune on the small high-consensus subset.**
→ **Transfer, directly:** pretrain on the 4,349 weak-labelled studies, then fine-tune on the 58
gold. **But with extreme care** — n=58, ~2× enriched, and it is simultaneously the only honest
validation set we own. Fine-tuning on it burns the instrument. The defensible version is
**cross-validated**: within each fold, fine-tune on that fold's gold rows and evaluate on the
held-out gold rows. See §8 bet F.

### 6.5 What did *not* transfer, and why — recorded so it is not re-investigated

| Idea | Verdict |
|---|---|
| 3-D CNN / nnU-Net end-to-end | **Rejected.** RSNA-2025 7th needed 4× A100 × 4.5 days. Our cap is 9 h on a T4 |
| Heavy multi-model ensembles (RSNA-2023 1st ran ~20 models) | **Rejected for the efficiency track, deferred for the main track.** They had 3× A6000 locally |
| TensorRT conversion | **Adapt, don't copy.** Use ONNX Runtime / `torch.compile` + fp16 on Kaggle |
| Horizontal-flip augmentation (used by almost every non-medical solution) | **Forbidden here.** It swaps medial and lateral, corrupting 5 of 12 labels. §10 |
| ImageNet-style aggressive colour jitter | **Reinterpret.** Colour jitter is meaningless on MRI, but its *purpose* — breaking scanner fingerprint — is exactly what §4.5 demands. Use intensity/gamma/noise/bias-field instead |

---

## 7. Literature, anatomy, and external assets

### 7.1 Which plane shows which finding — the anatomical basis for per-label slot weighting

Standard MSK radiology practice **[C — multiple independent radiology references]**:

| Plane | Primary role |
|---|---|
| **Sagittal** | **Cruciate ligaments (ACL/PCL)**, menisci (anterior and posterior horns — the "bow-tie" sign), cartilage |
| **Coronal** | **Collateral ligaments (MCL/LCL)**, menisci (body), medial/lateral compartment alignment and cartilage |
| **Axial** | **Patellofemoral joint** (→ PF OA), joint effusion extent, soft tissue, **Baker's cyst** (popliteal fossa) |

Minimum protocol for ACL imaging is T2 or PD-fat-sat in 2–3 orthogonal planes, plus at least one
T1 sagittal or coronal. **[C]**

Mapping that onto our 12 labels gives a **prior on which slot each label should attend to**:

| Label | Primary plane | Secondary |
|---|---|---|
| ACL | Sagittal | Coronal, Axial |
| MCL | **Coronal** | Axial |
| Medial / Lateral Meniscus | Sagittal | **Coronal** |
| Medial / Lateral OA | Coronal | Sagittal |
| PF OA | **Axial** | Sagittal |
| Effusion | Axial | any |
| Synovitis | Axial (fluid-sensitive) | Sagittal |
| Baker's | **Axial** | Sagittal |
| Contusion | any **fluid-sensitive** | — |
| Fracture | any | Sagittal, Coronal |

Note `Contusion` — bone-marrow oedema is essentially **invisible on structural sequences and obvious
on fluid-sensitive ones**. The fluid-sensitive bit is not decoration; for two labels it is the whole
signal.

**But do not hard-code this prior.** A 2025 study on the MRNet dataset built `TripleMRNet` (a CNN
over all three planes with adaptive average pooling) and trained all **seven** plane combinations.
**The three-plane model won on ACL** (accuracy 0.925, sensitivity 0.944, specificity 0.909, F1
0.919), and the paper's explicit headline was *"the unexpected importance of the axial plane"* — a
plane that conventional teaching treats as secondary for ACL. **[C]** Earlier radiology work
likewise found axial MRI contributes real independent information for **meniscal** tears. **[S]**

→ **Design conclusion:** use the table above to **initialise** attention, and let the model **learn**
the weighting. Hard-coding "ACL = sagittal only" throws away the axial signal that paper found.
This is exactly what a **learned target-specific attention pooling over slots** gives you (§8 E) —
prior as initialisation, evidence as the final arbiter.

### 7.2 MRNet — the reference benchmark for this exact clinical task **[C]**

Stanford, 1,370 knee MRI exams; 1,104 (80.6 %) abnormal, **319 (23.3 %) ACL tears**, **508 (37.1 %)
meniscal tears**. The MRNet model runs a CNN per series and combines sagittal T2 / coronal T1 /
axial PD predictions with **logistic regression**.

| Task | AUC (internal validation) |
|---|---|
| Abnormality | 0.937 (95 % CI 0.895–0.980) |
| **ACL tear** | **0.965** (0.938–0.993) |
| **Meniscal tear** | **0.847** (0.780–0.914) |

External validation matters for us: MRNet trained on Stanford sagittal T2 reached **0.824** on an
external set with no additional training, versus **0.911** when trained on that external data.
**That ~0.09 drop is the domain-shift tax**, measured, on this exact anatomy — an independent
confirmation of the site-leakage concern in §4.5.

**Calibration this gives us:**
- **ACL at ~0.96 is achievable** with strong labels. Our best public number for ACL is far below it,
  so ACL has headroom.
- **Meniscus at ~0.85 is hard even with strong labels.** The reported 0.476 for Medial Meniscus
  (§5.4) is therefore a *pipeline* failure (resolution/sampling), not a task ceiling.
- MRNet is a **binary abnormality** task with **clean radiologist labels**; ours is 12 severity-
  thresholded labels from weak text supervision. **Do not expect to match 0.96.** But a macro-AUC
  target in the **0.90–0.94** band is consistent both with MRNet's numbers and with the observed
  public leaderboard.

Other knee-MRI method papers worth knowing: **ELNet** (efficiently-layered network — explicitly
lightweight, relevant to §9) and **SB-SSL** (slice-based self-supervised transformers for knee
abnormality classification). **[S]**

### 7.3 DINOv2 and foundation-model backbones in medical imaging **[C]**

The public baseline uses **DINOv2-small (ViT-S/14)**, so this evidence matters:

- **Fine-tuning beats frozen features on medical data.** DINOv2 "consistently provides significant
  boosts when transferred to medical tasks **through unfreezing weights** during fine-tuning."
  A frozen encoder excels on natural-image-like tasks, but **domain shift to MRI limits it** relative
  to task-specific pretrained models.
- The public baseline's configuration — **last 6 blocks unfrozen, `lr_backbone = 8e-6`,
  `lr_head = 1e-3`** — is exactly this "partially unfrozen, discriminative LR" recipe. It is
  well-founded, not arbitrary.
- **LoRA / BitFit on a frozen DINOv2 give further gains updating <1 % of parameters** — directly
  relevant to a 9-hour budget and to the efficiency track.
- **Resolution governs transfer**: recent work on DINOv3 in chest radiography found **resolution
  scaling drives transfer performance**. That converges with the Nyquist argument in §3.6 from a
  completely different direction. **Two independent lines of evidence say resolution is the lever.**
- ViT patch size constrains input: ViT-S/**14** wants multiples of 14 (224, 336, 448 all qualify —
  convenient).

→ **Conclusion:** DINOv2-small partially unfrozen is a sound default; **but run the resolution study
(224 vs 336) before the backbone study.** Both the Nyquist argument and the DINOv3 resolution result
say resolution dominates architecture here, and §5.4's two below-chance labels are the predicted
symptom.

### 7.4 External datasets — catalogued, but **blocked** pending a host ruling

⚠️ **Do not build on any of these until the host rules** on whether click-through-licence datasets
count as "freely and publicly available" under the competition's external-data rule. **[S/?]**
Catalogued here so the work is ready if it clears.

| Dataset | Content | Relevance | Access |
|---|---|---|---|
| **MRNet** (Stanford) | 1,370 knee MRI exams; abnormal / ACL / meniscus labels; sag T2, cor T1, ax PD | **Highest.** Same anatomy, same planes, 3 of our 12 label concepts | Click-through research agreement. Train+val public, **test withheld** |
| **KneeMRI** (Rijeka, Croatia) | 917 exams, **ACL condition** labels, Siemens Avanto 1.5T | ACL only, single scanner — but a clean, well-characterised extra site | Public; also mirrored on Kaggle |
| **SKM-TEA** (Stanford) | 155 quantitative DESS knee scans; raw k-space, DICOM, **6-tissue segmentations, 16 pathology bounding boxes** | **Best source of anatomical segmentation labels** — could train a knee-structure localiser for a Stage-1 crop | Click-through |
| **fastMRI+** | **13 study-level labels + 16,154 bounding boxes** across 22 pathology categories, knee and brain | Pathology bounding boxes on knee MRI; closest thing to localisation ground truth | Click-through (fastMRI agreement) |
| **OAI** (Osteoarthritis Initiative) | Huge longitudinal knee cohort with **Kellgren–Lawrence / MOAKS OA grading** | **Directly relevant to our 3 OA labels**, which are the severity-graded ones | Registration required |

**If the ruling permits them, the highest-value use is not more training data — it is
segmentation/localisation labels.** SKM-TEA and fastMRI+ could train the Stage-1 anatomy localiser
that every prior RSNA winner had and that we otherwise lack (§6.3). That is a structural advantage,
not an incremental one.

### 7.5 Report labeling — the established methods to build on **[C]**

This is a solved-ish problem in English chest radiology, and the methods port:

- **CheXpert labeler** — rule-based pattern matching extracting 14 observations from free-text
  reports, formalising **mention / negation / uncertainty**. The canonical design, and the one the
  public baseline reimplements for knees.
- **NegBio** — extends NegEx using **universal dependencies and subgraph matching**; handles
  negation *and* uncertainty. More robust than regex for long clauses.
- **CheXbert** — distils a rule-based labeler into BERT using a small expert-annotated set, beating
  the rules it learned from. **This is our template**: rules → labels on 4,349 reports → train a
  multilingual encoder on those → the encoder generalises past the rules' gaps. Our 58 gold studies
  are the expert-annotated set that makes this legitimate.
- **Multilingual precedent exists** — CheXpert has been ported to **German** and **Brazilian
  Portuguese**, with domain-adapted LLMs the current approach. Our 12 languages are harder than any
  single port, but the method is proven.
- **CheX-GPT** uses zero-shot LLM labels as distant supervision to fine-tune a BERT labeler.
  ⚠️ For us this must use **open weights only** — §1.6.

→ **Practical ladder, cheapest first:** rules (works today, no GPU) → open-weights multilingual LLM
labeling on Kaggle (rules-safe) → distil both into a multilingual encoder (e.g. `multilingual-e5`,
XLM-R) trained on agreement, with **disagreement between labelers used as an uncertainty weight**
rather than forced to a hard label. §8 bet F.

---

## 8. Technical thesis

The bets, ranked by expected value per unit of effort. Each states what it is, why we believe it,
what would falsify it, and what it costs.

### Tier 1 — do these regardless of anything else

#### A. Severity-graded soft targets, not binary presence extraction

**What.** Extract an **ordinal severity score** per label from the report, map it to a soft target in
[0,1] approximating *P(two MSK radiologists would call this positive)*, and train with
soft-target BCE. Two extractors, not one — **magnitude vocabulary** for the six "how much?" labels,
**categorical vocabulary** for the six "what kind?" labels (§4.4).

**Why we believe it.** The rubric is severity-thresholded and the host said so explicitly (§2.4);
AUC is rank-only so granularity is free score (§4.3); and it has been measured — **+0.0464 macro AUC,
11 of 12 labels improved, P(Δ>0)=99.1 %** on the 58 gold studies (§4.3).

**Design rules, learned from others' bugs (§5.5, §5.6):**
- **Severity carries the rank estimate; `confidence` carries the uncertainty. Never conflate them.**
  Putting P(positive | unmentioned) into the severity value inverts evidence and cost one team
  0.121 AUC on Synovitis alone.
- **Test negation before pathology keywords**, or every negated mention scores positive.
- **"Not mentioned" is conditional on report completeness.** Confident 0 in a long structured report;
  soft and down-weighted in a 30-word one. Otherwise you inject site-correlated label bias (§3.3).
- Normalise Unicode carefully: Turkish dotless ı, Greek tonos, Cyrillic, the MICRO SIGN U+00B5.
- Clean the `intact9xintact4cm` substitution artifacts before any numeric parsing (§3.3).
- Map explicit grading scales: `grade/grado/graad/grad/derece/evre/βαθμός/степен`, **Outerbridge**
  and **ICRS** 1–4 for cartilage. The rubric's ">50 % thickness" is **Outerbridge 3–4**.

**Falsified if:** the severity extractor does not beat presence across ≥8 of 12 labels on the 58.
**Cost:** zero GPU. Pure text, runs on a laptop in seconds. **Do this first.**

#### B. Laterality canonicalisation

**What.** Resolve the knee side per study — DICOM `Laterality` → report first line → geometry sign
rule at |x| ≥ 20 mm — and mirror every study to one canonical side.

**Why.** 5 of 12 labels are side-specific; 50 % of studies have no `Laterality` tag; getting it wrong
scrambles half the label set. Geometry is 98.5 % accurate at the |x| ≥ 20 mm threshold (§3.7).

**Note this is defensive, not differentiating** — the public baseline already handles it. But it is
non-negotiable, and the three-source cascade with disagreement down-weighting is better than what
the baseline does.

**Implementation subtlety:** because the head pools over slices order-invariantly, only **in-plane**
mirroring matters. **Coronal and axial slices get horizontally flipped for the non-canonical side;
sagittal is left unchanged.** **[S]**

**Cost:** one header pass. **Falsified if:** report-side and geometry agree <90 % overall.

#### C. Site-grouped cross-validation, reported alongside random, from run one

**What.** `GroupKFold` on `language | manufacturer | model` (75 groups, max 5.8 %, 27 groups ≥50).
**Log both grouped and random val AUC every epoch, plus the gap.**

**Why.** Measured: random folds inflate by **+0.053** from headers alone and **+0.136** once a CNN
sees pixels (§4.5). Every architectural decision made on random-fold CV is a coin flip.

**And treat the gap as a first-class metric, not a diagnostic footnote.** Grouped CV is pessimistic
(train and test share the same 16 sites), random is optimistic; neither is the target. The **gap**
measures how much the model leans on scanner rather than anatomy, and shrinking it is an objective
in its own right.

**Cost:** free. **Falsified if:** the gap is near zero on a properly trained model — which would be
good news and would mean the augmentation work in D is unnecessary.

### Tier 2 — high expected value, needs measurement

#### D. Resolution, and anti-site augmentation — the two largest measured losses

**D1 — 336 px.** At `CROP_MM` 130, 224 px gives 0.58 mm/px against a ~1 mm meniscal tear; **336 px
gives 0.387 mm and clears Nyquist** (§3.6). The prediction is specific and testable: **menisci and
OA improve, Effusion (already 0.922, a large finding) does not.** Two below-chance labels in §5.4
were exactly the two fine-detail ones. Independent support: DINOv3 chest-radiograph work found
resolution scaling governs transfer (§7.3).
**Consider per-label resolution** — Effusion and Baker's are large findings and fine at 224; the
efficiency track will punish 336 everywhere.

**D2 — break the scanner fingerprint.** The +0.136 gap is the largest single measured loss in this
competition. The model reads noise texture, reconstruction kernel and native resolution. Counter with:
- **intensity / gamma / noise / bias-field augmentation** (the MRI analogue of colour jitter);
- **random-resized-crop**, to break the native-resolution signature;
- **earlier stopping** — grouped val plateaus at epoch ~6, gold peaks at ~9, and everything after is
  site memorisation;
- optionally **domain-adversarial training** on the site label (gradient reversal). Higher risk.
- **CutMix**, which worked at p=0.5–1.0 for the RSNA-2023 2nd place (§6.4).

**D3 — sagittal slice sampling.** On sagittal the menisci sit near the **ends** of the stack; a
(0.18, 0.82) band with 12 slices may sample straight past them. Widen the band and raise the slice
count on sagittal slots. **Cheap to test and it is the rival hypothesis to D1** — run both, they are
separable by whether Effusion moves.

#### E. Per-label slot attention, initialised from anatomy

**What.** Attention-pool over the 6 plane × sequence slots with a **learnable query per label**,
initialised toward the anatomical prior in §7.1, with the presence mask applied.

**Why.** The metric is **macro**, so a rare label is worth as much as a common one, and a single
shared pooling under-serves the rare ones. Anatomically, PF OA is axial, cruciates sagittal,
collaterals coronal — a shared mean pool wastes capacity averaging over slots that cannot see the
finding. This is the direct analogue of the RSNA-2023 2nd place's **organ-conditioned pooling with
independent per-organ logits** (§6.4), which is one of the best-validated ideas in the prior art.

**Initialise from the prior; do not hard-code it** — the TripleMRNet result (§7.1) found the axial
plane carries unexpected ACL signal, and hard-coding would discard it.

**Cost:** small parameter count, negligible runtime. **Falsified if:** learned attention converges to
uniform.

### Tier 3 — worth testing, lower confidence

**F. Multi-labeler disagreement as an uncertainty weight, plus gold fine-tuning.**
Build ≥2 independent labelers (rules + open-weights multilingual LLM on Kaggle), and where they
disagree, **down-weight the study in the loss** rather than forcing a hard label. Then the HMS
two-stage pattern (§6.4): pretrain on the 4,349 weak, fine-tune on gold — **but cross-validated
within folds**, because n=58 is simultaneously the only honest validation set we own. Burning it
leaves us flying blind.

**G. Auxiliary text distillation — privileged information.**
The report is available at train time and absent at test time: a textbook privileged-information
setup. Train the image encoder to **also** predict a frozen multilingual sentence embedding of the
report (auxiliary regression head, dropped at inference). This transfers signal the 12 binary labels
throw away, at **zero inference cost**. Higher risk, higher ceiling, and nobody else will bother.

**H. Metadata head as a low-weight ensemble member.**
0.598 site-grouped AUC from headers alone is not nothing. Blended at low weight it may add a few
thousandths, essentially free at inference. **But it is precisely the site-memorisation channel we
are trying to suppress** — blend at low weight and watch the grouped/random gap.

**I. A Stage-1 anatomy localiser.** Every prior RSNA winner had one (§6.3) and we do not. Blocked on
the external-data ruling (§7.4) unless we hand-annotate a few hundred slices ourselves — which the
mammography 1st place showed is enough (521 images for their YOLOX). **If the ruling opens
SKM-TEA / fastMRI+, this becomes Tier 1.**

**J. The Efficiency Track as a co-equal target.** See §9. It is the best risk-adjusted prize here,
and a single submission can win both.

### 8.1 What we deliberately are NOT doing

| Not doing | Why |
|---|---|
| 3-D CNN / nnU-Net | 4× A100 × 4.5 days for 7th place elsewhere; we have 9 h on a T4 (§6.5) |
| Horizontal flip augmentation | Corrupts 5 of 12 labels (§10) |
| Forking the public baseline wholesale | 0.891 is where ~everyone already is; it inherits one label extractor's ceiling (§5.1) |
| Tuning anything on the 58 gold studies | n=58, ~2× enriched, all have ≥1 positive. Direction only (§3.2) |
| Sending report text to a hosted LLM API | Rule 4.b risk, unruled (§1.6) |
| Chasing the public LB | ~390 studies; the private 70 % pays (§10) |

---

## 9. The Efficiency Track

### 9.1 Why it is the best risk-adjusted prize in this competition

**$18,000 across three places [S], and far fewer teams will contest it.** Most of the field is
optimising accuracy alone, so the efficiency leaderboard is thin. And critically: **a single
submission can win both tracks**, so this is not a fork in the road — it is a second lottery ticket
on the same work.

### 9.2 The metric, and its exchange rate **[S — verify before relying on it; §11]**

```
Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400
```

minimised, where `Benchmark` = the sample-submission AUC (0.5) and `maxAUC` = the best private-LB
score. With `maxAUC ≈ 0.95` the denominator is ≈ −0.45, so:

```
Efficiency ≈ −2.22 × AUC + Runtime / 32400
```

**The exchange rate: 0.01 AUC ≙ 720 seconds (12 minutes) of runtime.**

Worked example:

| Submission | AUC | Runtime | Efficiency | Winner |
|---|---|---|---|---|
| Fast | 0.900 | 20 min | **−1.963** | ✅ |
| Heavy | 0.920 | 3 h | −1.711 | |

The fast model wins decisively. **Using the full 9 hours costs 1.0 on this metric — equivalent to
giving away 0.45 AUC.** Almost no accuracy gain is worth the full budget.

⚠️ Note the metric depends on `maxAUC`, **which is not known until the competition ends**. As the top
score rises the denominator shrinks and accuracy gets *relatively* more valuable. The exchange rate
above assumes 0.95; at 0.92 it is 0.01 AUC ≙ ~840 s, at 0.97 ≙ ~630 s. **Same conclusion in every
case: be fast.** Kaggle publishes `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` — read it and
confirm the exact formula (§11).

### 9.3 The design that follows

Target **15–30 minutes total runtime**, and note from §3.8 that **decode is nearly free** (~3–4 min
for 1,300 studies on 4 processes), so almost the entire budget is model time.

- small backbone (EfficientNet-B0 / ConvNeXt-nano / DINOv2-small with LoRA);
- **modest resolution** — 224 where the label allows it; reserve 336 for the fine-detail labels only,
  or drop it entirely for this submission;
- fewer slices per slot (6–8 rather than 16);
- **no TTA, no multi-fold ensembling** — these are pure runtime with modest AUC return;
- fp16 + `torch.compile`, or ONNX Runtime. The mammography 1st place used TensorRT for exactly this
  (§6.4);
- batch the whole study; overlap DICOM decode with GPU compute.

**Log wall-clock runtime on every single run from day one.** It is half the metric and it cannot be
reconstructed afterwards. Forgetting is the single cheapest way to forfeit this prize.

---

## 10. Risk register

| Risk | Cost if ignored | Guard |
|---|---|---|
| **Random K-fold instead of site-grouped** | **+0.053 to +0.136 phantom AUC** — measured, twice | Group on `language\|manufacturer\|model`; report both every run (§8 C) |
| **Horizontal flip augmentation** | Silently corrupts 5 of 12 labels | **Never flip.** Not in train, not in TTA. Put an assert in the transform pipeline |
| **Ignoring laterality** | Medial/lateral inverted on ~half the studies | 3-source cascade + canonicalisation (§8 B) |
| **Fixed-pixel resize** | Anatomy scale varies **5.14×** | Crop to constant mm (130), then resize (§3.6) |
| **Soft-label collapse** | A whole training run wasted; OOF pred std ~0.05, AUC ≈ 0.5 | Log per-label prediction std every epoch; fail loudly (§5.5) |
| **Negation tested after pathology keywords** | Every negated mention scores positive | Test negation first; unit-test `"meniscus: no tear"` (§5.6) |
| **Treating `""` as present** | 20.9 % vs 50.7 % missing laterality; 83 vs 25 bilateral studies | Treat blank strings as null everywhere (§5.6) |
| **Tuning on the 58 gold studies** | Overfits an enriched n=58; destroys the only honest validation set | Direction check only; never a scoreboard (§3.2) |
| **Report text → hosted LLM API** | **Possible disqualification** (Rule 4.b, unruled) | Open weights, local or on Kaggle only (§1.6) |
| **External data before the host rules** | **Possible disqualification** | Blocked, not deprioritised (§7.4) |
| **Compressed transfer syntax in hidden test** | Inference notebook crashes; submission scores 0 | Ship `pylibjpeg`/`gdcm`; test the fallback path (§3.8) |
| **Forgetting to log runtime** | Forfeits the Efficiency Track | Log wall-clock on every run (§9.3) |
| **Overfitting the public LB** | Public is ~390 studies; private 70 % pays | Trust grouped CV; spend submissions sparingly |
| **Spending GPU quota on file I/O** | Burns the scarcest resource | Cache builds and labeling on **CPU sessions** — they draw no GPU quota |
| **Selecting a P100 on Kaggle** | Session dies at the first convolution: `CUDA error: no kernel image is available` — current Kaggle PyTorch ships no Pascal (sm_60) kernels. Failure appears only once training starts | Pin **T4** (`"machine_shape": "NvidiaTeslaT4"`). Turing sm_75, and it has tensor cores the P100 lacks **[S]** |
| **Thresholding the submission to binary** | Destroys rank information under AUC | Submit continuous probabilities (§2.6) |

---

## 11. Open questions

**This is the day-1 checklist.** Every item converts an [S] fact to [M], or answers a [?]. Items 1–9
run on a **Kaggle CPU session** (zero GPU quota) in about 30 minutes total — `src/phase0_verify.py`
runs them all and prints a pass/fail table against the [S] values asserted in §3.

### Must verify before building anything

| # | Question | How | Changes what |
|---|---|---|---|
| 1 | Are the counts right — 4,407 studies / 24,371 series / **58 gold**? | `pd.read_csv('train.csv')`, count non-null label rows | Everything. If gold ≫ 58 the whole weak-supervision framing weakens |
| 2 | **What is the exact evaluation metric text?** | Competition Evaluation tab | Macro-AUC assumed; a weighted or partial AUC changes per-label strategy |
| 3 | **What is the exact Efficiency metric?** | `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` | The §9.2 exchange rate |
| 4 | **Where is the leaderboard now?** (§5.1 is ~5 weeks stale) | Leaderboard tab | The target, and whether 0.891 is still the fork cluster |
| 5 | Has the host ruled on **external data** and on **hosted LLM APIs**? | Discussion search | Unblocks §7.4 (potentially Tier-1) and settles §1.6 |
| 6 | Is `Fluid_Sensitive` still perfectly correlated with `Fat_Suppression`? | cross-tab `train_series.csv` | 6 slots vs 12 |
| 7 | Is `Laterality` missing on ~50 %, **counting blanks as missing**? | value counts on the header scan | Whether the geometry fallback is needed |
| 8 | Are all training series really uncompressed Explicit VR LE? **And what about test?** | `TransferSyntaxUID` counts | Cache-build budget **and** inference runtime (§9) |
| 9 | Do `PatientID`s repeat? | `nunique` vs row count | Whether folds must group on patient too |

### Resolve in the first week

| # | Question | Changes what |
|---|---|---|
| 10 | Does severity beat presence on the 58, **in our own implementation**? | Whether bet A is the thesis or a footnote |
| 11 | Does 336 px fix Medial Meniscus and MCL, while leaving Effusion flat? | Whether the §5.4 failure is resolution or sampling (D1 vs D3) |
| 12 | What is the **true prevalence** of each label? The 58 are ~2× enriched and all have ≥1 positive | Loss weighting; sanity-checking the labeler's output distribution |
| 13 | How large is the grouped-vs-random gap **in our pipeline**? | How much anti-site augmentation is worth |
| 14 | Does `SeriesDescription` beat the CSV columns for slot assignment? | Slot-assignment rule (expected: no — §3.5) |
| 15 | Does the report-side extractor reproduce ~98.8 % accuracy where it fires? | Confidence in the laterality cascade |
| 16 | How many test studies actually have all 6 slots? | How much the presence mask matters at inference |

### Structural unknowns

| # | Question | Note |
|---|---|---|
| 17 | Are train/public/private prevalences different? | The host reportedly warned they are **[S]**. If so, rank-based AUC protects us but calibration-based tricks will not transfer |
| 18 | Is the hidden test set from the **same 16 sites**? | Believed yes **[S]** — which is why grouped CV is pessimistic rather than the target (§4.5) |
| 19 | Are any of the 58 gold studies also in the public LB split? | Would make the 58 less independent than assumed |

---

## 12. Provenance and confidence ledger

Where each block of this document comes from, so its weight can be judged.

| § | Content | Source | Confidence |
|---|---|---|---|
| 1 | Environment, network, credential test | **Our own commands in this session** | **[M]** |
| 2.1–2.3, 2.7 | Competition identity, labels, timeline, prizes, format | WebSearch over Kaggle/RSNA/press + both competitor repos | **[C]** |
| 2.4–2.5 | Label criteria, ground-truth construction, host Q&A | One competitor repo quoting Kaggle discussions we could not read | **[S]** |
| 2.6 | Macro ROC-AUC, submission format | WebSearch + both repos | **[C]** |
| 3.1–3.2 | Counts: 4,407 / 24,371 / 58 / 4,349 | **Two independent repos agreeing** | **[C]** |
| 3.3–3.8 | Languages, slots, headers, scale, laterality, decode | One competitor repo's measured scan | **[S]** |
| 4.3 | Severity-vs-presence +0.0464 result | One repo's experiment, self-described as partly in-sample | **[S]** |
| 4.5 | Site-leakage +0.053 / +0.136 | Two separate experiments, both single-source | **[S]** |
| 5.1–5.3 | Leaderboard shape, public baseline design | One repo + WebSearch. **~5 weeks stale** | **[S]** |
| 5.4–5.6 | Training result, version history, bugs | One repo's experiment log | **[S]** |
| 6 | Prior competitions and winning architectures | **Winner repos cloned and read directly**, plus writeup summaries | **[M]/[C]** |
| 7.1 | Plane/anatomy mapping | Multiple independent radiology references | **[C]** |
| 7.1 | TripleMRNet axial-plane finding | Peer-reviewed study (QIMS 2025) via search summary | **[C]** |
| 7.2 | MRNet benchmark numbers | PLOS Medicine paper via search summary | **[C]** |
| 7.3 | DINOv2 in medical imaging | Multiple papers via search summaries | **[C]** |
| 7.4 | External dataset catalogue | Search summaries; **licence status unverified** | **[S]/[?]** |
| 7.5 | Report-labeling methods | Established literature (CheXpert/NegBio/CheXbert) | **[C]** |
| 8, 9, 10 | Thesis, efficiency analysis, risk register | **Our synthesis** over the above | **[I]** |
| 9.2 | The efficiency formula itself | One repo | **[S]** — verify (§11 #3) |

### Attribution

The two competitor repositories were the decisive sources and deserve naming. They are public work
by teams competing in this same competition; nothing here was obtained privately, and nothing from
them is reproduced as code in this repo:

- **`github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection`** — the measured 819 k-file
  scan, label-criteria transcription, fold analysis, laterality validation, severity-extractor
  experiment, and training log. **Most of §3, §4 and §5 rests on this.**
- **`github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection`** — independent corroboration of the
  counts, the versioned baseline scores, and the V04 design notes.

**Bear in mind they are competitors.** Their factual measurements are probably sound — and are
corroborated where it matters (§3.9) — but their *strategic* framing is theirs, not necessarily
right, and not necessarily disinterested. §11 exists so we stand on our own measurements as fast as
possible.

See [docs/SOURCES.md](docs/SOURCES.md) for the complete URL index.
