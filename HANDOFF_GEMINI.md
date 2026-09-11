# Handoff to the next agent — RSNA 2026 Knee Abnormality Detection

**For the human, first, in three lines.** Give the next agent *both* documents: the
`RSNA_KNEE_AGENT_PROMPT.md` directive **and** this file. The directive is a good document and
you should keep using it. This file is the reconciliation between it and the repository it will
be working in, plus four corrections — one of which is a disqualification risk you need to read
before the agent does anything.

Everything below is addressed to the incoming agent.

---

## PART A — WHAT YOU ARE BEING HANDED

You have two inputs that were produced independently and have never been reconciled:

1. **The directive** (`RSNA_KNEE_AGENT_PROMPT.md`) — an operating discipline. Gates, evidence
   protocol, leak tests, adversarial review. It is written in reaction to a prior agent that
   shipped a model trained on Gaussian noise and reported a fabricated CV of 0.7936.
2. **This repository**, branch `claude/pensive-euler-lt2rpc` — a research dossier and tested
   Phase-0 scaffolding. 18.5k words of documentation, 1,338 lines of Python.

**They are not rival plans. They are different halves of the same job.** The directive says
*how to work and what to prove*. This repo says *what is already known and what code already
exists*. There is no architectural disagreement between them — see Part B. There are exactly
four points of friction — see Part C.

### What is in this repo

| File | Lines | What it is |
|---|---|---|
| `WORKING_NOTE.md` | 1,557 | The dossier. Every claim carries a confidence tag: `[M]`easured by us, `[C]`orroborated by 2+ independent sources, `[S]`econd-hand single source, `[I]`nference, `[?]`open |
| `PLAN.md` | 244 | Campaign plan for the 41 days remaining, ~180 GPU-hours |
| `docs/SOURCES.md` | 107 | Every URL consulted, marked read-directly / search-summary-only / blocked |
| `src/phase0_verify.py` | 473 | **Header-only DICOM scan that grades itself against the documented values.** This is the directive's Stage 1, already written |
| `src/report_labeler.py` | 529 | Deterministic multilingual regex extractor. This is Stage 3 Track A, already written. 20 unit tests pass |
| `src/score_labelers.py` | 205 | Dependency-free AUC + bootstrap CI; scores severity-vs-presence labelling on the gold studies. This is Gate G3's scoring half |
| `src/folds.py` | 131 | Grouped CV on `language\|manufacturer\|model`, with a random-fold control. This is T5's fold builder |

### What is NOT in this repo, and this matters

**Nothing was ever trained, and no competition data was ever touched.** `kaggle.com`,
`api.kaggle.com` and `kaggleusercontent.com` were blocked by egress policy — 403 at the proxy
CONNECT stage, before authentication — so credentials could not and did not help. The box also
had no GPU and ~30 GB of writable disk against a ~570 GB dataset.

So: **there are no fabricated metrics in this repo, because there are no metrics in this repo.**
Every empirical number in `WORKING_NOTE.md` is tagged as somebody else's measurement, and
`src/phase0_verify.py` exists specifically to replace those tags with your own.

All code here was smoke-tested against a synthetic dataset shaped like the real one, and the AUC
implementation was checked against scikit-learn over 500 tie-heavy cases. That is the *only*
validation it has. It has never seen a real DICOM.

---

## PART B — WHERE THE DIRECTIVE AND THIS REPO ALREADY AGREE

Do not re-derive these. They are settled from both directions.

| Point | Directive | This repo |
|---|---|---|
| Reports exist only in train; never a model input | D3 | §4, the "two instruments" framing |
| Verify every inherited dataset fact by measurement | D5, Stage 1 | §11 day-1 checklist + `phase0_verify.py` |
| Never random KFold; group by site/scanner | T5, Stage 4 | §4.5, `folds.py` |
| Report grouped *and* random CV so inflation is measured | Stage 4 | §4.5 (reports a +0.053–0.136 gap, second-hand) |
| Soft/graded labels, not binary | Stage 3 | §4 — AUC is rank-only, so grading is strictly better |
| Deterministic multilingual regex extractor, ~9 languages | Stage 3 Track A | `report_labeler.py` |
| 6 slots = plane × fat-sat, key-padded not zero-filled | Stage 2 | §3.5, §8 |
| 130 mm physical crop, 336 px, 16 slices, 0.12–0.88 band | Stage 2 / Part 5 | §3.6 (incl. the Nyquist argument for 336) |
| Geometric laterality, 20 mm deadband, plane-dependent flip | Stage 2 | §3.7, `phase0_verify.py:geometric_side()` |
| No horizontal-flip augmentation | implied by the flip canonicalisation | §3.7 — **5 of 12 labels are side-specific**, stated explicitly |
| DINOv2, last-N-blocks unfrozen, LR 8e-6 / 1e-3 | Stage 4 / Part 5 | §8 Tier-1 |
| 12-query cross-attention head with slot embedding | Stage 4 | §8 |
| Rank-space blending; drop members correlating >0.98 | Stage 5 | §8 |
| Efficiency track is under-contested and worth real effort | Stage 6 | §9, with the exchange rate worked out |

**A caution about that agreement.** It is weaker evidence than it looks. The directive's Part 5
says it read the 0.935 public notebook directly; this repo's §3 and §8 came from two competitor
GitHub repos that are very likely forks or descendants of that same notebook. Two documents
agreeing because they drank from the same river is not two independent confirmations. Treat the
shared pipeline as *the strong public baseline to beat*, not as proven-optimal.

---

## PART C — FOUR CORRECTIONS TO THE DIRECTIVE

Ranked by stakes, highest first.

### C1. The hosted-LLM label pipeline is a disqualification risk, and you are the hosted LLM

Directive Part 2 says: *"using a commercially hosted LLM offline to convert reports into labels
… is believed permitted. **Verify this yourself.**"* Stage 3 Track B then builds a whole label
track on it and says to "parallelize across fast-model workers."

`WORKING_NOTE.md` §1.6 reaches the opposite default, from Competition Rule 4.b (Data Security),
which plausibly forbids sending Competition Data to any non-participating party:

> **Do not paste report text into any hosted model — this assistant included — and do not call a
> commercial LLM API on report text.** Use open-weights multilingual models (Qwen3, Gemma,
> multilingual-e5) run locally or inside a Kaggle notebook.

**Read the sharp edge here.** You are a commercially hosted model. If the human pastes report
text into your context so you can extract labels, *that is the exact act under question* — not a
preparation for it. The directive's own "verify this yourself" instruction has to be satisfied
**before** Track B begins, not during it, because by then the data has already left.

This is also the one item the directive itself routes to the human: Part 8 escalation trigger #1
is "rules ambiguity with compliance risk (LLM usage…)". So:

- **Default to NO** until a host ruling is found. Search the competition discussion for host
  posts on LLM APIs and on external data (§11 item #5).
- The fallback the directive names is already built: `src/report_labeler.py`, Track A,
  unambiguously compliant, zero external calls.
- If you want a second extractor for the Stage 3 reconciliation, use an **open-weights** model
  run locally or inside the Kaggle kernel. That gives you Track B's cross-validation value with
  none of its exposure.
- The cost of complying is near zero. The cost of not complying is the whole competition.

Note for the record: no competition data was handled in the session that produced this repo, so
prior exposure from this lineage is zero.

### C2. The disputed numbers are second-hand, not fabricated — and you should still measure them

Directive D5 says the counts (4,407 studies, 58 gold, 24,371 series, 50.7% missing laterality)
*"come from a source that has already been shown to contain fabricated content."*

**The instruction is right. The characterisation conflates two different things.**

The fabrication was a *training result* — a CV of 0.7936 produced by a model fed noise. The
counts above came from a different place: two public competitor repositories that scanned the
real data independently and agree with each other on every one of them (`WORKING_NOTE.md` §3.9):

| Fact | Repo A | Repo B | Agree |
|---|---|---|---|
| Train studies | 4,407 | 4,407 | ✅ |
| Train series | 24,371 | 24,371 | ✅ |
| Gold-labelled studies | 58 | 58 | ✅ |
| Report-only studies | 4,349 | 4,349 | ✅ |

And now the honest half: **I cannot prove they are right.** I never touched the data. Under this
repo's own confidence legend those are `[C]`, never `[M]`, and the interpretive claims layered on
top (severity thresholds, site-leakage magnitudes, extractor accuracies) are single-source `[S]`
and individually marked. So D5's practical conclusion holds in full — measure them.

**What follows from this is a shortcut, not an argument.** Do not start Stage 1 from scratch.
`src/phase0_verify.py` *is* Stage 1 with the D5 audit already compiled in: it performs the
header-only scan, writes `series_headers.parquet`, and grades what it measures against an
`ASSERTED` dict of the documented values with per-fact tolerances:

```python
ASSERTED = {
    "train_studies": (4407, 0.00), "train_series": (24371, 0.00),
    "gold_studies": (58, 0.00), "fluid_fatsat_offdiag": (0, 0.00),
    "laterality_missing_frac": (0.507, 0.10), "pixelspacing_ratio_p99_p1": (5.14, 0.40),
    "fov_p1_mm": (130, 0.10), "uncompressed_frac": (1.00, 0.02), ...}
```

Run it on a Kaggle CPU session — **no GPU quota, ~30 minutes** — and its pass/fail table *is*
`docs/MEASURED_FACTS.md` for Gate G1. Every disagreement is flagged automatically, and where it
disagrees, your measurement wins and the note is wrong. That is what the file was built for.

Then apply the same skepticism symmetrically: **the directive is also a document of unverified
provenance.** Its "best public notebook ≈ 0.935, leader ≈ 0.952" and its Part 5 description
deserve the same treatment under its own D5. They are probably fresher than this repo's §5.1
leaderboard snapshot, which is roughly five weeks stale and should not be used to set targets —
but check the live leaderboard yourself before trusting either.

### C3. "Delete all prior code" does not mean this repo

Directive Part 10 #2 names `train_dinov2_3x3.py`, `ensemble_submission_pipeline.py`,
`rsna_knee_dinov2_ensemble_submission.py`, and `dinov2_multimodal_fold*.pt`. **Verified: none of
those files exist on this branch, and there are no checkpoints of any kind here.** That
instruction targets a different codebase. Quarantine it there; leave this repo intact.

There is also no training script here at all, contaminated or otherwise. Stage 4 is yours to
write from zero.

### C4. Read the clock before you plan

The directive states no deadline. There are **41 days** (to 2026-10-22) and roughly 180 GPU-hours
of Kaggle quota. `PLAN.md` is built around that budget and targets a first real submission by
day 7.

Stage 0's gate — break the pipeline seven ways, prove each test catches its target — is correct
and you should do it. Budget it honestly: it is a day or two, and it comes out of the 41.

---

## PART D — YOUR FIRST DAY

Ordered. Items 1–4 need no GPU.

1. **Read `WORKING_NOTE.md` §0 (handoff contract) and §11 (open questions).** §0 tells you what
   not to redo. §11 is the day-1 verification checklist the directive's Stage 1 asks for.
2. **Resolve C1 before anything else touches report text.** Search the competition discussion for
   host rulings on hosted LLM APIs and external data. Write the finding to `docs/DECISIONS.md`
   with the URL. If unresolved, Track A only.
3. **Re-read the live competition pages** — leaderboard, exact Evaluation text, exact Efficiency
   formula, rules. §11 items #2–#5. The §5.1 snapshot is stale; do not plan against it.
4. **Run `python src/phase0_verify.py` on a Kaggle CPU session.** Its output table is Gate G1.
   Anything it flags, the note is wrong and you correct the note.
5. **Build Stage 0.** See Part E — this is the real gap.
6. **Run `python src/score_labelers.py`.** It scores graded-severity labels against binary-presence
   labels on the gold studies and prints a PASS/FAIL verdict. Note the directive's own warning
   here, which this repo shares: **per-finding AUC on 58 studies has an enormous error bar.** The
   script reports bootstrap CIs for that reason. Treat it as a direction indicator, not a number.

---

## PART E — THE GENUINE GAP: STAGE 0 DOES NOT EXIST HERE

This is the directive's single best contribution and the thing this repo most lacks.

`src/phase0_verify.py` verifies **facts about the data**. It does not verify **the pipeline**.
There is no `tests/leak_suite.py`, no label-shuffle test, no constant-input test, no grep guard.
Nothing in this repo would have caught the prior agent's failure, because nothing in this repo
watches a training path — there is no training path yet.

**Build Stage 0 exactly as the directive specifies, T1–T7, including the seven induced failures.**
Do not skip the induced-failure step. A test suite that has never caught a bug is decoration, and
the directive is right that this is the stage whose absence caused the 0.5.

Two additions worth making while you are there:

- **T2 (label shuffle) is the load-bearing one.** Run it on every architecture change, not just
  once. It is the single test that catches the exact prior failure.
- The directive's T6 submission checks (per-column std floor, unique-value count, no pair of
  findings correlating at 1.0) would have caught the constant submission. Add one more: assert the
  submission's study IDs match `sample_submission.csv` as a **set**, not just in count.

---

## PART F — PRE-CLEARED GREP-GUARD HITS

Directive T7 says to grep `src/` for forbidden constructs and justify or remove every hit. Run it
yourself, but here is the current result so you do not delete working code:

```
src/folds.py:86           rng = np.random.default_rng(seed)     # seeded shuffle, bin-packing tiebreak
src/folds.py:88           load = np.zeros(n_splits, dtype=int)  # integer load counter, not image content
src/folds.py:99           rng = np.random.default_rng(seed)     # random_folds — the deliberate CV control
src/folds.py:116          rng = np.random.default_rng(0)        # self-test synthetic groups, __main__ only
src/score_labelers.py:69  rng = np.random.default_rng(seed)     # bootstrap resampling of studies
src/phase0_verify.py:303  rng = np.random.default_rng(0)        # subsampling which series to scan
```

**Six hits, none in a model data path** — there is no model data path yet. All are seeded RNG for
fold assignment, bootstrap CIs, or sampling *which real files to read*. None generates anything a
model consumes as a feature. `folds.py:99` in particular is the random-KFold control the directive
itself asks for in Stage 4; it is supposed to be there.

When you write Stage 2 and Stage 4, this clean sheet is what you are protecting.

---

## PART G — WHAT I COULD NOT DO, SO YOU DO NOT OVER-TRUST THIS REPO

Stated plainly, because the directive is right that inherited documents deserve suspicion:

- **No competition data was ever accessed.** Every data fact in `WORKING_NOTE.md` §3 is somebody
  else's measurement, tagged as such.
- **Nothing was trained. No metric here is an experimental result.** There is no CV score, no LB
  score, no OOF file.
- **The code has never seen a real DICOM.** It passes its own unit tests against synthetic inputs
  shaped like the real thing. That is meaningfully less than working.
- **§5.1's leaderboard snapshot is ~5 weeks stale.**
- **§3's framing comes from a competitor's repository.** Their measurements are corroborated;
  their strategic conclusions are theirs, and are not necessarily right.
- **External datasets (MRNet, OAI, fastMRI+, SKM-TEA) are treated as blocked pending a host
  ruling**, not merely deprioritised. See §7.4 and C1.

The directive's standard is the right one and this repo does not meet it, by construction: nothing
here is `[M]` because nothing here could be. Your first job is to convert it. The tooling for that
conversion is already written and waiting.

One last thing, and it applies to you specifically. Speed makes the evidence protocol matter
*more*, not less. A fast model asked for a status report will produce a fluent, plausible,
well-formatted one whether or not the run happened. That is precisely how the last agent's 0.7936
got into a log. D1 is the antidote: a number without a log pointer is `UNVERIFIED`, and
`UNVERIFIED` is an acceptable answer.
