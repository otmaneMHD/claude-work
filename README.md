# RSNA 2026 Knee Abnormality Detection — campaign workspace

Kaggle: [`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
· 12 binary findings per knee MRI study · **macro ROC-AUC** · code competition, ≤9 h, internet off
· final submission **2026-10-22** · $77,000 including a **separate Efficiency Track**

---

## Read in this order

| Doc | What it is |
|---|---|
| **[AGENT_DIRECTIVE.md](AGENT_DIRECTIVE.md)** | **The master prompt. Start here.** Gates, evidence protocol, leak suite, build order, definition of done — and Part 11, what this repo is and how much of it to trust. |
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

Then work to **[AGENT_DIRECTIVE.md](AGENT_DIRECTIVE.md)**, which is the master prompt: the gates,
the evidence protocol and the seven-test leak suite that this repo's own scaffolding does *not*
provide. Two things in it are load-bearing and easy to skim past:

- **Do not send report text to a hosted LLM** pending a host ruling on Competition Rule 4.b
  (Part 2). If you are a hosted model, that includes your own context.
- **Nothing in this repo is measured.** Every data fact is second-hand and tagged. Converting it
  to measurement is your first job, and `src/phase0_verify.py` is the tool that does it.
