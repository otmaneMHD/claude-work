# Source index

Everything consulted in the 2026-09-11 research session, what it gave us, and whether it was read
directly or only through a search summary.

**Access legend:** ✅ read directly (cloned or fetched) · 🔍 search summary only · ⛔ blocked by the
egress policy in this session (§1.2)

---

## 1. The competition itself — ⛔ all blocked, known via search summaries only

| URL | What it holds |
|---|---|
| <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection> | Overview |
| …/data | Data description, file layout, column definitions |
| …/rules | **Rule 4.b (Data Security)** — the hosted-LLM question (§1.6) |
| …/leaderboard | Current standings. **§5.1 is ~5 weeks stale — re-read this first** |
| …/discussion/733343 | Host: challenge overview + **the 12 label criteria** (§2.4) |
| …/discussion/733826 | Host Q&A: labels are image-derived and authoritative (§2.5) |
| …/discussion/733517 | Metadata-shortcut probe: 0.6515 random vs 0.5981 grouped (§4.5) |
| …/discussion/733652 | Rules clarification thread: external data + LLM APIs |

## 2. Public Kaggle notebooks — 🔍 known by reference only

| Notebook | Note |
|---|---|
| `pilkwang/rsna-knee-baseline-v1` | **The reference baseline.** 221 votes, 0.809; forks cluster at 0.891 (§5.2) |
| `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` | **Kaggle-official Efficiency leaderboard.** Read it to confirm §9.2 |
| `debugendless/rsna-knee-abnormality-detection-baseline` | Second public baseline |
| `romantamrazov/rsna-knee-dinosaur-v2` | DINOv2 variant |
| `xiaoleilian/rsna26-knee-eda` | Public EDA |
| `mpwolke/rsna-knee-abnormality-detection-2026-dcm` | DICOM handling |
| `riachk/rsna-knee-abnormality-detection-tool` | Published Kaggle Model |

## 3. Competitor repositories — ✅ cloned and read in full. **The decisive sources.**

### `github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection` ✅

1,315 lines of strategy docs + a 586-line multilingual labeler + EDA scripts + Kaggle kernels.
**Most of §3, §4 and §5 of the working note rests on this.**

| File | What it gave us |
|---|---|
| `docs/STRATEGY.md` | Label criteria, ground-truth construction, the 6-slot finding, `CROP_MM`, efficiency arithmetic, ranked ideas |
| `docs/FINDINGS.md` | **The measured 819,078-file header scan** — laterality, folds, physical scale, decode cost, slice counts, report language stats, severity-extractor results |
| `docs/EXPERIMENTS.md` | **The train-v1 result** (§5.4) — per-label gold AUC, the +0.136 site gap, the two below-chance labels |
| `docs/ROADMAP.md` | Their 75-day phase plan and GPU budget |
| `docs/PLATFORM.md` | Kaggle sizing, cache arithmetic, **the P100 warning** |
| `docs/RESEARCH_AGENDA.md` | Their Phase-0 question list — the basis for our §11 |
| `docs/DAY1.md` | Their trap list |
| `src/report_labeler.py` | Design reference: clause splitting, negation/normality/hedge lexicons, magnitude and grade scales across 9 languages |

### `github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection` ✅

Independent corroboration (§3.9) plus a versioned baseline with logged Kaggle scores.

| File | What it gave us |
|---|---|
| `README.md` | Counts (4,407 / 24,371 / 819,078 / 58), the 12 label names, V01–V04 scores |
| `src/v04-...ipynb` | DINOv2-S + laterality normalisation + target-specific attention pooling; the **plane-specific mirroring** rule (§8 B) |

## 4. Prior-competition winner solutions

| Repo / writeup | Place | Access | What it gave us |
|---|---|---|---|
| `github.com/Nischaydnk/RSNA-2023-1st-place-solution` | **1st**, RSNA 2023 abdominal | ✅ | The canonical 2.5D recipe: 96 slices → (32,3,H,W), CoaT-Lite/EffNetV2-S @384, GRU head, GroupKFold; **targets scaled by organ visibility** |
| `github.com/TheoViel/kaggle_rsna_abdominal_trauma` | **2nd**, RSNA 2023 abdominal | ✅ | **Organ-conditioned pooling**, **independent per-organ logits**, CutMix p=0.5–1.0, MaxViT-tiny@512, Ranger |
| `github.com/dangnh0611/kaggle_rsna_breast_cancer` | **1st**, RSNA 2023 mammography | ✅ | YOLOX-nano ROI detector trained on **521 images**, ConvNeXt-small on crops, **TensorRT inference** |
| `github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution` | 7th/1147, RSNA 2025 | ✅ | nnU-Net ResEnc-M 3-D, **200×160×160 mm ROI crop**; and the **4× A100 × 4.5 days** cost that rules 3-D out for us |
| …/rsna-2024-lumbar…/writeups/avengers-1st-place-solution | **1st**, RSNA 2024 lumbar | 🔍 | Two-stage CenterNet keypoints (EffNet-B6+FPN) → per-level crops; **centre vs side classifiers**; pseudo-labels; **hand-corrected annotations** |
| …/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution | **1st**, RSNA 2025 | 🔍 | *"Location-Aware Aneurysm Detection via Vessel-ROI Masking"* — localise first |
| …/hms-harmful-brain-activity-classification/discussion/492560 | **1st**, HMS 2024 | 🔍 | Two-stage: pretrain on noisy labels → fine-tune on high-consensus subset |
| RSNA 2022 cervical spine, 1st | 🔍 | | 15 slices/vertebra, neighbours **+ segmentation masks as channels**, EffNetV2-S/ConvNeXt + LSTM |

## 5. Literature — 🔍 search summaries (arxiv.org, pubmed and most journal hosts were ⛔)

| Work | What it gave us |
|---|---|
| **MRNet** — Bien et al., *PLOS Medicine* 2018; `stanfordmlgroup.github.io/projects/mrnet/` | The benchmark: ACL 0.965, meniscus 0.847, abnormality 0.937; the **0.09 external-validation domain-shift tax** (§7.2) |
| **TripleMRNet** — *Quantitative Imaging in Medicine and Surgery*, 2025 (PMID 40606344) | All 7 plane combinations on MRNet; three-plane best for ACL (acc 0.925); **"the unexpected importance of the axial plane"** (§7.1) |
| *Meniscal Tears: Role of Axial MRI Alone and in Combination with Other Imaging Planes*, AJR | Axial contributes independent meniscal signal |
| **ELNet** (arXiv 2005.02706) | Efficiently-layered network for knee MRI — relevant to §9 |
| **SB-SSL** (arXiv 2208.13923) | Slice-based self-supervised transformers for knee abnormality classification |
| *Comparative Analysis of ImageNet Pre-Trained Models and DINOv2 in Medical Imaging* (arXiv 2402.07595) | Fine-tuning beats frozen DINOv2 on medical data (§7.3) |
| *Resolution scaling governs DINOv3 transfer in chest radiograph classification* (arXiv 2510.07191) | **Resolution dominates architecture** — converges with the Nyquist argument (§3.6) |
| **CheXpert labeler**, **NegBio**, **CheXbert** (arXiv 2004.09167), German CheXpert (arXiv 2306.02777) | The report-labeling method ladder (§7.5) |
| MRI knee protocol references (Radiopaedia/IMAIOS/Kenhub/mrimaster) | The plane → finding mapping (§7.1) |

## 6. External datasets — 🔍 catalogued, **licence status unverified, use blocked pending host ruling** (§7.4)

| Dataset | Where |
|---|---|
| MRNet | `stanfordmlgroup.github.io/competitions/mrnet/` |
| SKM-TEA | `aimi.stanford.edu/datasets/skm-tea-knee-mri` |
| fastMRI+ | *Scientific Data* (2022), `nature.com/articles/s41597-022-01255-z` |
| KneeMRI (Rijeka) | also mirrored on Kaggle |
| OAI | Osteoarthritis Initiative, registration required |

## 7. Press and announcements — 🔍 (rsna.org and auntminnie.com were ⛔)

| Source | What it confirmed |
|---|---|
| `rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge` | The official challenge page |
| `rsna.org/news/2026/august/ai-challenge-knee-mri` + press release 2669 | **$77,000**, efficiency award, 5,000+ exams, **12 languages, 16 sites, 5 continents**, 22 Oct deadline |
| `auntminnie.com/.../15831808/` · `radiologybusiness.com` · `runtimewire.com` | Independent confirmation of the above |
| `x.com/kaggle/status/2085036812922134737` | Kaggle's announcement — the 12-finding framing |
