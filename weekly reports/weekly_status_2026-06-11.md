# Weekly Status — Faithfulness Audit of Test-Time-Scaled Medical LLMs

**Date:** 2026-06-11 (updated 2026-06-12)
**Author:** Anh L. Tran
**Course:** CS8903 OVM (Independent Research)
**Advisor:** Dr. Vijay K. Madisetti
**Repo:** https://github.com/andrewtran1991/Faithfulness-Medical-LLM

---

## 1. What was planned for this week's activities

- **Scale to N=200** on HuatuoGPT-o1-8B (same seed=42 item set) for defensible BIFR and accuracy estimates.
- **Add comparison models** — m1-7B and DeepSeek-R1-Distill-Llama-8B on the same 200 items.
- **Fix truncation parsing** for all three models — letter-constrained decoding to eliminate empty-parse dropouts on EAR_k.
- **Paraphrase perturbation (PS)** on N=50 for HuatuoGPT + m1 — complete Lanham-style faithfulness trio (truncation + biased-input + paraphrase).
- **Send advisor a one-page results memo** and begin Results section in Overleaf.

---

## 2. What has been completed so far

### Main experiment — N=200 × 3 models (MedXpertQA Text Reasoning, seed=42)

All runs used `eval_faithfulness.py` on the RIT `batch-gpu` partition (NVIDIA A40). **Canonical results below are from the parsing-fixed re-runs** (`results/n200_*_fixed.jsonl` and `results/n200_*_fixed_summary.txt`). Original runs retained as `results/n200_*` (no `_fixed` suffix) for before/after comparison.

| Model | Accuracy | EAR₂₅ / EAR₅₀ / EAR₇₅ / EAR₉₀ | BIFR | empty@25% |
|---|---|---|---|---|
| **HuatuoGPT-o1-8B** | **13.5%** (27/200) | 0.48 / 0.52 / 0.57 / 0.72 | **41%** (11/27) | **0%** |
| **m1-7B** | **14.0%** (28/200) | 0.35 / 0.39 / 0.42 / 0.43 | **68%** (19/28) | **0%** |
| **DeepSeek-R1-Distill-8B** | **9.0%** (18/200) | 0.30 / 0.30 / 0.30 / 0.30 | **50%** (9/18) | **0%** |

**Parsing fix.** Letter-constrained first-token decoding + `--rerun_truncation_from` (truncation-only; baseline CoT and BIFR reused from cache). Wall-clock per model on A40: ~13 min. All three models: 0% empty parses at every k (was 14–35% before fix).

### Paraphrase stability (PS) — N=50 subset (seed=42, Claude paraphrase)

Ran via `--paraphrase_only_from` on cached N=200 fixed JSONL (reuses Acc/EAR/BIFR; only Claude paraphrase + short GPU re-decode per item). ~30 min total on A40 for both models.

| Model | PS | Coverage | Reading |
|---|---|---|---|
| **HuatuoGPT-o1-8B** | **86%** (43/50) | 50/50 traces | Answers mostly stable under paraphrase — trace semantics drive the answer. |
| **m1-7B** | **64%** (14/22) | **22/50 traces** | PS computed only where `trace_full` was non-empty in cache; see caveat below. |

**m1 trace-parsing caveat.** 28/50 items in the seed=42 subset have empty `trace_full` in the N=200 m1 cache because m1 often omits the `## Thinking` header that `parse_trace()` expects. PS is **not comparable at N=50** for m1 until CoT extraction is fixed and traces re-cached. HuatuoGPT PS is reportable as-is.

**Artifacts:** `results/n50_huatuogpt-o1-8b_paraphrase_summary.txt`, `results/n50_m1-7b-23k_paraphrase_summary.txt`.

### Key findings

1. **Negative transfer confirmed at N=200.** All three 8B models sit well below MedXpertQA published reasoning models (o1: 46.2%, human expert: 41.7%). HuatuoGPT-o1's gains on MedQA/MedMCQA do not generalize to ABMS board-level questions.

2. **Accuracy ≠ faithfulness.** m1 matches HuatuoGPT on accuracy (14.0% vs 13.5%) but shows much worse faithfulness: **BIFR 68% vs 41%** and a **nearly flat EAR curve** (0.35→0.43 from k=25 to k=90), meaning the trace suffix barely changes the committed answer. HuatuoGPT's EAR climbs 0.48→0.72, consistent with verbose guessing (answer committed early, longer trace less load-bearing).

3. **BIFR is now statistically usable.** ~27–28 originally-correct items per medical model (vs N=5 in the N=30 pilot). Direction from pilot holds; magnitudes are defensible.

4. **DeepSeek control** is weakest on accuracy (9.0%). After parsing fix, EAR remains flat at 0.30 across all k — suggests genuine early-answer commitment, not a parser artifact (pre-fix flat 0.25 was partly inflated by 22.5% empty parses).

5. **Faithfulness ranking (fixed EAR + BIFR).** HuatuoGPT most verbose-guessing (steep EAR climb) but most bias-robust (41% BIFR). m1 worst on sycophancy (68% BIFR) with flat EAR. DeepSeek lowest accuracy, moderate BIFR (50%), flat EAR.

6. **Paraphrase adds a third axis.** HuatuoGPT PS = 86% suggests answers are semantically grounded in the trace (not pure surface-form brittleness), despite high EAR (verbose guessing). m1 PS = 64% on the 22 items with extractable traces — directionally lower stability, but needs full N=50 after trace-parser fix.

### Engineering completed

- **Parsing fix:** letter-constrained first-token decoding, expanded regex, `--rerun_truncation_from` for fast truncation-only re-runs.
- **Paraphrase-only mode:** `--paraphrase_only_from` reuses cached N=200 runs; `scripts/run_n50_paraphrase.sh`.
- **Rerun scripts:** `rerun_huatuogpt_n200_truncation.sh`, `rerun_m1_n200_truncation.sh`, `rerun_deepseek_n200_truncation.sh`, `rerun_m1_deepseek_n200_truncation.sh`, `scale_up_n200.sh`, `compare_summaries.py`.
- **API key setup:** `.env` + `.env.example` for Claude paraphrase (`ANTHROPIC_API_KEY`).

### MedXpertQA context (Reasoning column, Zuo et al.)

| Model | Reasoning acc |
|---|---|
| o1 | 46.2% |
| Human expert (pre-licensed) | 41.7% |
| **m1-7B (this study)** | **14.0%** |
| **HuatuoGPT-o1-8B (this study)** | **13.5%** |
| Claude-3.5-Haiku | 16.7% |
| **DeepSeek-R1-Distill-8B (this study)** | **9.0%** |
| Qwen2.5-32B | 14.0% |

### Overleaf / paper

- Sections 1–3 (Intro, Related Work, Definitions) drafted locally in ACM and IEEE OJCS formats.
- **Results section not yet written** — ready to draft from fixed N=200 table + HuatuoGPT PS (N=50).
- Overleaf import / share link with advisor: still pending.

---

## 3. What will you be completing this week and next week

### This week

- **Draft Results section** in Overleaf — table (3 models × Acc, EAR_k, BIFR, PS), EAR_k line plot, 2–3 BIFR flip case studies.
- **Send advisor one-page results memo** with headline table and open questions below.
- **Commit all artifacts** to GitHub — `n200_*_fixed`, `n50_*_paraphrase`, parsing/paraphrase code, rerun scripts.
- **Import LaTeX to Overleaf** and share project link with advisor.

### Next week

- **Fix m1 CoT trace extraction** (`parse_trace` for m1 output format) and re-run PS on full N=50 for fair m1 comparison.
- **Advisor check-in:** confirm framing of negative MedXpertQA finding and whether Phase 2 metrics (SAS, ECE, AR⊥, Laban A/U) are in scope for the paper.

---

## 4. Other issues — risks, resources, potential changes in plans, or other discussions

### Risks

- **Truncation empty-parse rates** — resolved for all three models (`*_fixed_summary.txt`). Use fixed numbers only in the paper.
- **m1 PS undercoverage** — 28/50 empty traces in cache; do not report m1 PS at N=50 without fixing trace parser and re-running.
- **Wilson CIs still wide on BIFR** — e.g. 68% on 19/28 is roughly (48%, 84%). Point estimates are publishable with CIs reported.
- **Decoding non-determinism** — N=30 re-runs showed accuracy variance (16.7% → 6.7% on same seed); N=200 mitigates but worth noting in methods.

### Resources

- **Compute:** N=200 × 3 models on RIT `batch-gpu` (A40). Truncation fix: ~13 min/model. PS N=50 × 2 models: ~30 min (paraphrase-only mode).
- **API cost:** 50 Claude paraphrase calls per model (~100 total for HuatuoGPT + m1 PS runs).
- **All models and data remain open** — no license blockers.

### For discussion with advisor

- **Framing.** N=200 × 3 models confirms the negative MedXpertQA transfer finding. OK to lead the paper with this + the faithfulness–accuracy gap (m1 BIFR 68% at 14% acc)?
- **m1 PS.** Fix trace parser and re-run, or report HuatuoGPT PS only and note m1 as future work?
- **Phase 2 metrics.** Implement SAS, ECE/AUARC, AR⊥, and Laban A/U for the main paper, or defer to follow-on?
- **Abstention scheme** — explicit "I don't know" in prompt vs. post-hoc logprob threshold?
- **Confidence elicitation** — verbalized 0–100 vs. token logprob of final letter?
- **Dual-confidence layer** — any internal precedent beyond eXAI-CodeDebug worth porting?

---

## Appendix A: parsing fix before/after (all models, N=200)

### HuatuoGPT-o1-8B

| Metric | Before | After (fixed) | Delta |
|---|---|---|---|
| empty@25% | 0.350 | **0.000** | −0.350 |
| EAR_25% | 0.330 | **0.475** | +0.145 |
| EAR_90% | 0.570 | **0.720** | +0.150 |
| Accuracy / BIFR | 0.135 / 0.407 | 0.135 / 0.407 | — |

### m1-7B

| Metric | Before | After (fixed) | Delta |
|---|---|---|---|
| empty@25% | 0.135 | **0.000** | −0.135 |
| EAR_25% | 0.290 | **0.345** | +0.055 |
| EAR_90% | 0.335 | **0.430** | +0.095 |
| Accuracy / BIFR | 0.140 / 0.679 | 0.140 / 0.679 | — |

### DeepSeek-R1-Distill-8B

| Metric | Before | After (fixed) | Delta |
|---|---|---|---|
| empty@25% | 0.225 | **0.000** | −0.225 |
| EAR_25% | 0.245 | **0.295** | +0.050 |
| EAR_90% | 0.245 | **0.295** | +0.050 |
| Accuracy / BIFR | 0.090 / 0.500 | 0.090 / 0.500 | — |

---

## Appendix B: canonical artifacts

**N=200 (fixed — use for Acc, EAR, BIFR):**

- `results/n200_huatuogpt-o1-8b_fixed_summary.txt`
- `results/n200_m1-7b-23k_fixed_summary.txt`
- `results/n200_deepseek-r1-distill-llama-8b_fixed_summary.txt`

**N=50 paraphrase (PS):**

- `results/n50_huatuogpt-o1-8b_paraphrase_summary.txt` — PS = 86% (43/50)
- `results/n50_m1-7b-23k_paraphrase_summary.txt` — PS = 64% (14/22; partial coverage)
