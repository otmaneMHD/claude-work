# Campaign plan — 2026-09-11 → 2026-10-22

**41 days to final submission. 34 days to the entry/team-merger deadline (15 Oct).**

> This plan is deliberately *not* the 75-day campaign the competitor repo laid out from 8 Aug.
> We are entering with **roughly half that runway** and nothing trained. The plan below is
> compressed accordingly: it front-loads the cheap, high-certainty wins, gets a submission in early
> to close the loop, and refuses to spend GPU on anything that has not first won on a subset.

Read [WORKING_NOTE.md](WORKING_NOTE.md) first. Every §-reference below points into it.

---

## 1. Objectives, in priority order

1. **Ship a working end-to-end submission by day 7.** Not to compete — to close the loop and
   establish that our pipeline produces a scored number. Everything else is worthless until this
   exists.
2. **Beat the public fork cluster (~0.891, §5.1) on our own grouped CV *and* on the LB.**
3. **Contest the Efficiency Track (§9).** Best risk-adjusted prize in the competition, thin field,
   and **the same submission can win both tracks**.
4. Be able to **explain and reproduce** the solution: winners must submit training code, weights, a
   method description and a video. A solution you cannot defend is one you cannot claim.

## 2. The currency is GPU-hours, not days

Kaggle gives ~**30 GPU-h/week**. Six weeks ≈ **180 GPU-hours** for the entire campaign — about 40 %
of what the 75-day plan assumed. Allocation:

| Phase | Days | GPU-h | Share |
|---|---|---|---|
| 0 — Verify and label | 11–14 Sep | **0** | CPU only |
| 1 — Foundations + first submission | 15–18 Sep | 20 | 11 % |
| 2 — Label quality frozen | 19–24 Sep | 25 | 14 % |
| 3 — Vision model | 25 Sep – 8 Oct | 60 | 33 % |
| 4 — Specialisation + efficiency build | 9–15 Oct | 40 | 22 % |
| 5 — Ensemble + robustness | 16–20 Oct | 25 | 14 % |
| 6 — Freeze + deliverables | 21–22 Oct | 5 | 3 % |
| Reserve | — | 5 | 3 % |

**Three rules that protect the budget:**

- **I/O never touches a GPU.** Cache builds, metadata scans and report labeling run on **CPU
  sessions**, which draw nothing from the GPU quota.
- **Nothing is promoted to a full run until it has won on a subset.** A 1,200-study / 1-fold /
  3-epoch probe costs ~30 min and kills most ideas.
- **Every run logs wall-clock runtime.** It is half the Efficiency metric and cannot be
  reconstructed afterwards (§9.3).

## 3. Platform

**Kaggle, for everything that touches a GPU or a DICOM.** Not negotiable, and not a preference:

- The data is **already mounted** at `/kaggle/input/rsna-knee-abnormality-detection/` — zero
  transfer against ~570 GB.
- It is a **code competition**: the final submission must be a Kaggle notebook regardless, so
  anything built elsewhere has to come home to score.
- This sandbox cannot help — `kaggle.com` is blocked, there is no GPU, and there are 30 GB of disk
  against a 570 GB dataset (§1.1, §1.7).

| Where | Accelerator | What |
|---|---|---|
| Local / this repo | none | Code authoring, label-extractor development, docs, git |
| Kaggle job 1 | **CPU (off-quota)** | Metadata scan → parquet of series facts |
| Kaggle job 2 | **CPU (off-quota)** | Pixel cache build → publish as a Kaggle Dataset |
| Kaggle job 3 | **CPU (off-quota)** | Open-weights multilingual LLM labeling pass (rules-safe: text never leaves Kaggle) |
| Kaggle training | **T4** | All model training against the cache |
| Kaggle submission | T4 | Inference notebook, internet off, ≤9 h |

⚠️ **Pin the T4.** A P100 session dies at the first convolution — current Kaggle PyTorch ships no
Pascal (sm_60) kernels, the GPU is still offered in the UI, and the failure only appears once
training starts, costing a queued session. Set `"machine_shape": "NvidiaTeslaT4"`. **[S]**

**Cache sizing** (`4407 × 6 slots × S slices × P² bytes`, uint8). Kaggle persists 20 GB from
`/kaggle/working`; `/kaggle/temp` gives ~60 GB of non-persistent scratch.

| P | S | Size | Fits 20 GB? |
|---|---|---|---|
| 224 | 12 | 15.9 GB | ✅ |
| 224 | 16 | 21.2 GB | 2 shards |
| **336** | **16** | **47.8 GB** | 3 shards |

**Decision: build at 224 px / 16 slices in two shards first** (Phase 1), because it unblocks
everything immediately and the decode cost is only ~15 min on 4 processes (§3.8). **Rebuild at
336 px in three shards as a second pass in Phase 3**, once the resolution hypothesis (§8 D1) has
been confirmed on a subset. Store `uint8`, not `float32` — intensity is already normalised to [0,1]
before quantisation, so 8 bits cost nothing a bilinear resize has not already cost, and the file is
a quarter the size.

---

## Phase 0 — Verify and label (11–14 Sep) · **0 GPU-hours**

**Nothing is trained. Nothing is cached. This phase exists because every number we hold is
second-hand (§0.4), and building on unverified facts is how campaigns die in week 5.**

| Job | Where | What |
|---|---|---|
| **0.1** | Kaggle **CPU** | Run `src/phase0_verify.py` — the §11 checklist. Prints a pass/fail table against every [S] value asserted in §3. **~30 min.** |
| **0.2** | Kaggle **CPU** | Header-only metadata scan (`stop_before_pixels=True`) over all series → one parquet row per series: plane, slot, slice count, `PixelSpacing`, geometry, laterality, scanner fingerprint, transfer syntax. **~10 min on 4 processes.** |
| **0.3** | Kaggle / local | Read the Evaluation tab, the Efficiency LB notebook, the current leaderboard, and search the discussions for host rulings on **external data** and **hosted LLM APIs** (§11 #2–#5) |
| **0.4** | local, CPU | Build the **report labeler**: presence extractor (control) + severity extractor (thesis), two vocabularies per §4.4. Score both on the 58 gold studies |

**Exit criteria — all four must hold:**

1. We can state, with our own numbers, what one training example looks like: how many slots, how
   many slices, what physical crop, canonicalised to which side.
2. The fold scheme is decided and the group sizes are measured.
3. **The severity thesis has survived first contact** — or been falsified. If severity does not beat
   presence on ≥8 of 12 labels, say so plainly and fall back to presence labels. **It is a bet, not
   a belief.**
4. We know whether external data and open-weights-vs-API labeling are permitted.

---

## Phase 1 — Foundations and first submission (15–18 Sep) · **~20 GPU-h**

| Step | Where | What |
|---|---|---|
| 1.1 | Kaggle CPU | **Pixel cache**: 224 px / 16 slices / 6 slots / `CROP_MM` 130, canonicalised. Two shards → Kaggle Datasets |
| 1.2 | Kaggle CPU | **Visual verification** — montages per slot for ~10 studies. Left and right knees must look identical in orientation; joint centred. **Do not skip this.** Geometry bugs are silent and poison everything downstream |
| 1.3 | local | Site-grouped CV harness on `language\|manufacturer\|model`; **both grouped and random val logged every epoch** |
| 1.4 | Kaggle T4 | First end-to-end model. Small backbone, 1 fold, short. Target < 30 min |
| 1.5 | Kaggle T4 | **First submission.** Close the loop |

**Exit criterion:** a scored submission from our own pipeline, and a logged runtime number.

⚠️ **Instrument for soft-label collapse from run one** (§5.5): log per-label prediction standard
deviation every epoch and fail loudly if it collapses toward the base rate. This costs nothing and
cost another team an entire training run.

---

## Phase 2 — Freeze the label set (19–24 Sep) · **~25 GPU-h**

This is where the competition is won or lost (§4.1).

| Step | What |
|---|---|
| 2.1 | Open-weights multilingual LLM labeling pass on a Kaggle CPU session (Qwen3 / Gemma / multilingual-e5). Report text never leaves Kaggle → rules-safe |
| 2.2 | **Three label sets compared head-to-head under identical training**: presence / severity-rules / LLM |
| 2.3 | Labeler **disagreement** used as a per-sample uncertainty weight, not forced to a hard label (§8 F) |
| 2.4 | **Hand-review** the ~100 studies where the labelers disagree most. The RSNA-2024 winners hand-corrected their annotations and it is very likely the highest-yield hour in this campaign (§6.4) |
| 2.5 | **The diagnostic:** compare grouped-CV score against LB score. A large gap means *labels* are the bottleneck; a small gap means *vision capacity* is |

**Exit criterion:** the label set is frozen, and we know which bottleneck we are fighting.

**Prior from §5.4(a):** the text labeler already beats the vision model by 0.117 on the 58 gold
studies, which says **vision capacity is currently the bottleneck**. If Phase 2 confirms that, cut
this phase short and move the hours into Phase 3. **Do not over-invest in labels out of attachment
to the thesis.**

---

## Phase 3 — Vision model (25 Sep – 8 Oct) · **~60 GPU-h** · the largest block

| Step | What | Predicted outcome |
|---|---|---|
| 3.1 | **Resolution study: 224 vs 336** (§8 D1). Rebuild cache at 336 in 3 shards | Menisci and OA improve; **Effusion does not** |
| 3.2 | **Sagittal slice sampling**: widen the band beyond (0.18, 0.82), raise slice count (§8 D3) | The rival hypothesis to 3.1 — separable by whether Effusion moves |
| 3.3 | **Anti-site augmentation** (§8 D2): intensity / gamma / noise / bias-field, random-resized-crop, CutMix, earlier stopping | The grouped-vs-random gap falls from +0.136 |
| 3.4 | **Per-label slot attention** initialised from the §7.1 anatomical prior (§8 E) | Rare and focal labels improve; learned attention ≠ uniform |
| 3.5 | Backbone comparison — **only after 3.1–3.4 are settled.** Changing it earlier confounds everything | DINOv2-S partially unfrozen is the default (§7.3) |
| 3.6 | Multi-fold, multi-seed to get a variance estimate | ±0.02 on gold is currently unresolvable |
| 3.7 | Auxiliary text-distillation head (§8 G) **if time allows** | Zero inference cost, higher ceiling |

**Exit criterion:** a single model that beats the public fork cluster on our own grouped CV *and*
on the LB.

**Order matters and is not arbitrary.** Resolution and sampling first, because §5.4 gives two
below-chance labels with a specific, testable diagnosis. Augmentation second, because the +0.136
site gap is the largest single measured loss. Architecture **last**, because it is the thing most
teams reach for first and the thing least likely to matter here.

---

## Phase 4 — Specialisation and the efficiency build (9–15 Oct) · **~40 GPU-h**

| Step | What |
|---|---|
| 4.1 | Split into label groups by what actually sees them: **big-fluid** (Effusion, Baker's), **fine-detail** (Meniscus ×2, OA ×3), **marrow** (Contusion, Fracture), **ligament** (ACL, MCL). Per-group resolution and slice budgets |
| 4.2 | **Per-label resolution**: 224 for large findings, 336 for fine ones. Cheaper than 336 everywhere and probably no worse |
| 4.3 | **Build the efficiency submission in parallel** (§9.3): small backbone, 6–8 slices, no TTA, no ensemble, fp16 + `torch.compile`, target **15–30 min** total |
| 4.4 | Handle the compressed-transfer-syntax fallback and test it (§10) |

**Exit criterion:** two candidate submissions — one maximal, one fast.

🔴 **15 Oct is the entry and team-merger deadline.** Nothing to do if competing solo, but it is a
hard gate — you cannot join or merge after it.

---

## Phase 5 — Ensemble and robustness (16–20 Oct) · **~25 GPU-h**

| Step | What |
|---|---|
| 5.1 | Ensemble across folds / seeds / resolutions. **Weight per label, not globally** — the metric is macro |
| 5.2 | Robustness checks: studies missing slots, the 25 bilateral studies, odd slice counts, unseen-scanner holdout |
| 5.3 | Optional low-weight metadata head (§8 H) — **watch the grouped/random gap when blending it** |
| 5.4 | **Resist LB-chasing.** Public LB is ~390 studies; the private 70 % pays. Trust grouped CV |

---

## Phase 6 — Freeze and deliver (21–22 Oct) · **~5 GPU-h**

- Select the two final submissions (accuracy + efficiency).
- Re-run the inference notebook end-to-end from a cold start, internet off, and confirm it completes
  inside 9 h with margin.
- Assemble winners' obligations **now, while the reasoning is fresh**: training code, weights, method
  description, video, public model release (reportedly due ~5 Nov).

---

## 4. Decision log — locked as of 2026-09-11

| # | Decision | Basis |
|---|---|---|
| 1 | Kaggle-only for GPU and DICOM work | §1.1, §1.7, code competition |
| 2 | `CROP_MM = 130`, cache at 224 px first, 336 px in Phase 3 | §3.6, cache sizing above |
| 3 | 6 plane × sequence slots with a presence mask | §3.4 |
| 4 | CSV columns primary for slot assignment; `SeriesDescription` auxiliary | §3.5 |
| 5 | Laterality: DICOM tag → report first line → geometry at \|x\| ≥ 20 mm; canonicalise | §3.7 |
| 6 | Bilateral (25 studies): simple rule, no dedicated workstream | §3.7 |
| 7 | Folds: `GroupKFold` on `language\|manufacturer\|model`; report grouped **and** random every run | §4.5 |
| 8 | No patient-level grouping (zero repeat `PatientID`s) | §4.5 |
| 9 | **No horizontal flip. Ever.** Train or TTA | §10 |
| 10 | Two extractors — magnitude for 6 labels, categorical for 6; soft targets + separate confidence | §4.4, §8 A |
| 11 | "Unmentioned" is soft and down-weighted in short reports | §3.3 |
| 12 | Target **both tracks** with one pipeline | §9.1 |
| 13 | 2.5D, not 3-D | §6.5 |
| 14 | External data and hosted-LLM labeling: **blocked pending host ruling** | §1.6, §7.4 |
| 15 | Submit continuous probabilities, never thresholded | §2.6 |

## 5. What would make us change course

| Signal | Response |
|---|---|
| Phase 0 shows gold studies ≫ 58 | The weak-supervision framing weakens; shift weight from labels to standard supervised training |
| Severity does not beat presence on ≥8/12 labels | Drop bet A to Tier 3, fall back to presence, reallocate to Phase 3 |
| 336 px does not fix the fine-detail labels | The failure is sampling (D3), not resolution. Cheaper, and good news for the efficiency track |
| Grouped/random gap stays >0.12 after augmentation | Escalate to domain-adversarial training on the site label |
| Host permits external data | **SKM-TEA / fastMRI+ become Tier 1** — they supply the Stage-1 anatomy localiser every prior RSNA winner had and we lack (§8 I) |
| Host forbids open-weights LLM labeling too | Rules-based extractor only. It already scores 0.791 on the 58 (§5.4a), so this is survivable |
| LB has moved far above 0.94 by Phase 3 | Re-weight hard toward the Efficiency Track, where the field is thin (§9.1) |
