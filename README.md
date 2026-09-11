# RSNA 2026 Knee Abnormality Detection — campaign workspace

Kaggle: [`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
· 12 binary findings per knee MRI study · **macro ROC-AUC** · code competition, ≤9 h, internet off
· final submission **2026-10-22** · $77,000 including a **separate Efficiency Track**

---

## Read in this order

| Doc | What it is |
|---|---|
| **[HANDOFF_GEMINI.md](HANDOFF_GEMINI.md)** | **Start here if you arrived with the `RSNA_KNEE_AGENT_PROMPT.md` directive.** Reconciles that directive with this repo: where they agree, the four places they clash, and what to do first. |
| **[WORKING_NOTE.md](WORKING_NOTE.md)** | **The exhaustive note.** Every fact, its source, its confidence, and the full audit trail of this session. **Read this first — it is the handoff.** |
| **[PLAN.md](PLAN.md)** | The campaign plan: phases, decisions, exit criteria, GPU budget, day-by-day schedule to 22 Oct |
| [docs/SOURCES.md](docs/SOURCES.md) | Every URL, repo and artifact consulted, with what it gave us |
| [src/](src/) | Runnable pipeline scaffolding — written to be pasted into Kaggle notebooks |

## State of the work, as of 2026-09-11

**Research phase complete. Nothing has been trained. No competition data has been downloaded.**

This session ran inside a sandbox whose egress policy **blocks `kaggle.com` entirely**, so the
~570 GB dataset could not be fetched and the Kaggle pages could not be read directly. Everything
in `WORKING_NOTE.md` was assembled from sources that *were* reachable — principally two public
GitHub repositories from teams already working this competition, which between them contain a
**measured 819 k-file scan of the real data**, plus winner-solution repositories from six prior
RSNA/medical-imaging competitions.

Read [WORKING_NOTE.md §1](WORKING_NOTE.md#1-session-audit) for the exact access audit before
concluding anything is missing. **Every data fact in these documents is marked with its provenance
and a confidence level.** Second-hand facts are labelled as such and each carries a cheap
verification command to run in the first Kaggle session.

## If you are a successor agent picking this up

Do **not** start by downloading the data or re-running the web research. Both are already
summarised. Start at [WORKING_NOTE.md §0](WORKING_NOTE.md#0-handoff-contract), which tells you
exactly what is established, what is assumed, and what the first three things to verify are.

If you were handed the separate `RSNA_KNEE_AGENT_PROMPT.md` directive alongside this repo, read
**[HANDOFF_GEMINI.md](HANDOFF_GEMINI.md)** first instead. The two documents are complementary —
the directive supplies the gates and evidence discipline this repo lacks, this repo supplies the
research and scaffolding the directive assumes you will build from scratch — but they conflict on
four points, one of which (sending report text to a hosted LLM) carries compliance risk and must
be settled **before** any report text is handled.
