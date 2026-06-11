# Weekly Status — Faithfulness Audit of Test-Time-Scaled Medical LLMs

**Date:** 2026-06-04
**Author:** Anh L. Tran
**Course:** CS8903 OVM (Independent Research)
**Advisor:** Dr. Vijay K. Madisetti
**Repo:** https://github.com/andrewtran1991/Faithfulness-Medical-LLM

---

## 1. What was planned for this week's activities

- **Reproduce baselines.** Run HuatuoGPT-o1-8B on MedXpertQA and confirm published numbers within a reasonable margin.
- **Define metrics.** Operationalize reasoning-faithfulness metrics in measurable form — formulas, decision rules, sample sizes.
- **End-to-end pilot.** Wire baseline + perturbations together and run on a small sample to validate the protocol.

---

## 2. What has been completed so far (past two weeks) and Overleaf updates

### Foundation built

- **Research proposal drafted.** *Beyond Final-Answer Accuracy: Auditing Reasoning Faithfulness in Test-Time-Scaled Medical LLMs.*
- **Faithfulness metric spec written.** 10 operational metrics: truncation EAR_k, paraphrase stability PS, biased-input flip rate BIFR, step-ablation SAS, ECE, selective AUARC, abstention rate AR_⊥, and the Laban aptitude/unreliability decomposition. See `faithfulness_metric_spec.md`.
- **Eval pipeline implemented.** `eval_faithfulness.py` runs the audit using HuggingFace `transformers` directly — no need to spin up the HuatuoGPT-o1 repo's sglang server for pilot-scale work.
- **Paper drafts (ACM + IEEE OJCS).** Sections 1 (Introduction), 2 (Related Work / Existing Work), and 3 (Definitions) drafted in both `acmsmall` and `ieeetj` formats.
- **Code + results published to GitHub.** Public repo at https://github.com/andrewtran1991/Faithfulness-Medical-LLM (README, metric spec, pilot memo, raw JSONL results).

### Pilot results — first published reading of HuatuoGPT-o1-8B on MedXpertQA Text Reasoning (N=30)

| Metric | Value | Reading |
|---|---|---|
| **Baseline accuracy** | **5/30 = 16.7%** | Below the pre-licensed human expert (41.7%) and below every reasoning model in Zuo et al. Table 5. |
| EAR_25 / EAR_50 / EAR_75 / EAR_90 | 0.50 / 0.50 / 0.57 / 0.63 | Half the items keep the same final answer when the CoT is truncated to its first 25%. EAR climbs only 13 pp from k=25 to k=90 — a faithful model would show a steep climb. **Strong verbose-guessing signal.** |
| **BIFR** | **3/5 = 60%** | Of the items the model got right, 3 flipped when prefixed with "I'm fairly sure the answer is \[wrong letter]." **Strong sycophancy signal**, with small-N caveat (5 originally-correct items). |

**Run details.** RunPod 1× RTX PRO 6000, ~22 min wall-clock, ~$0.77 of compute. Artifact: `results/pilot_huatuogpto1_8b.jsonl`.

### Pilot in context (MedXpertQA Text Reasoning subset)

| Model | Reasoning acc |
|---|---|
| o1 | 46.2% |
| Human expert (pre-licensed) | 41.7% |
| DeepSeek-R1 | 37.9% |
| o3-mini | 37.6% |
| LLaMA-3.3-70B (vanilla) | 23.9% |
| QwQ-32B-Preview | 18.7% |
| Claude-3.5-Haiku | 16.7% |
| **HuatuoGPT-o1-8B (this pilot)** | **16.7%** |
| Qwen2.5-32B | 14.0% |

**Takeaway.** HuatuoGPT-o1's +8.5 pp average gain on MedQA / MedMCQA / PubMedQA does *not* transfer to MedXpertQA-level difficulty. Training on MedQA-USMLE and MedMCQA likely doesn't generalize to ABMS specialty-board questions.

### Overleaf updates

LaTeX drafts exist locally:

- `ACM_paper/main.tex` — ACM (`acmsmall`)
- `IEEE_OJCS_paper/main.tex` — IEEE OJCS (`ieeetj`)

**Next:** 5-minute import to an Overleaf project; will share the project link with advisor this week. \[Replace this line with the Overleaf URL once imported.]

---

## 3. What will you be completing this week and next week

### This week

- **Fix parsing dropouts in the truncation perturbation.** 8/30 truncated outputs returned empty letters at k=25 (parser couldn't find a letter after the force-answer prefill). Bump `max_new_tokens` and expand the answer regex.
- **Re-run N=30 with the fix** to confirm EAR_k rises after the parser change.
- **Lock the next-pilot scope with the advisor** (model set, N, paraphrase-test on/off) — to pin down in this meeting.
- **Import LaTeX drafts to Overleaf** and share the project link.

### Next week

- **Scale to N=180 raw items on HuatuoGPT-o1-8B.** Yields ~30 originally-correct items → defensible BIFR estimate.
- **Add m1 and DeepSeek-R1-Distill-Llama-8B comparison runs** on the same items. Tests whether the underperformance is HuatuoGPT-specific. ≈ $15 compute total.
- **Add paraphrase perturbation (LLM-judged)** — completes the Lanham-style faithfulness trio (truncation + biased-input + paraphrase).
- **Send the advisor a one-page results memo** before the next check-in.

---

## 4. Other issues — risks, resources, potential changes in plans, or other discussions

### Risks

- **N=30 baseline is a small-N point estimate.** Wilson 95% CI on 5/30 is roughly (6%, 35%). The N=180 follow-up will tighten this dramatically.
- **60% BIFR is on N=5 originally-correct items.** Striking but the CI is too wide to defend; the direction is clear, the magnitude is not.
- **Parsing dropouts depress EAR readings.** Current EAR numbers underestimate verbose-guessing; the fixed re-run will show a higher number.

### Resources

- **Compute is not the bottleneck.** RunPod at $2.09/hr (1× RTX PRO 6000); full N=180 × 3 models ≈ $15 total.
- **All assets are open.** HuatuoGPT-o1, m1, MedXpertQA — no licenses needed.
- **Code and results are public on GitHub** for any collaborator to reproduce.

### For discussion with advisor

- **Framing of the 16.7% finding.** Is this a publishable negative finding ("o1-style medical training does not transfer to MedXpertQA-level difficulty") that we lead with now, or do we hold until the N=180 + m1 + DeepSeek-R1-Distill comparison confirms? Lean toward hold-and-confirm, but want your read.
- **Abstention scheme.** Explicit "I don't know" option in the prompt, or post-hoc logprob threshold?
- **Confidence elicitation.** Verbalized ("0–100") vs. token logprob of the final answer letter?
- **Dual-confidence layer precedent.** Any internal precedent beyond eXAI-CodeDebug (Dubey & Madisetti) worth porting from?
