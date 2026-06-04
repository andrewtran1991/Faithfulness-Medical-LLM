# Faithfulness-Medical-LLM

**Beyond Final-Answer Accuracy: Auditing Reasoning Faithfulness in Test-Time-Scaled Medical LLMs**

A research project that audits whether the accuracy gains of test-time-scaled medical LLMs (HuatuoGPT-o1, m1, o1) come from genuinely faithful reasoning or from spending more tokens before guessing.

**Author:** Anh L. Tran (Georgia Tech, CS8903 OVM)
**Advisor:** Dr. Vijay K. Madisetti

---

## Headline finding (pilot, N = 30)

First published reading of **HuatuoGPT-o1-8B on MedXpertQA Text Reasoning**:

| Metric | Value | Reading |
|---|---|---|
| Baseline accuracy | **5/30 = 16.7%** | Below the pre-licensed human expert (41.7%) and every reasoning model in Zuo et al. (2025). |
| EAR_25 | 0.50 | Half the items keep the same final answer when the CoT is cut to its first 25%. |
| EAR_50 | 0.50 | Same. |
| EAR_75 / EAR_90 | 0.57 / 0.63 | EAR climbs only 13pp from k=25 to k=90 — a faithful model would show a steep climb. |
| BIFR | **3/5 = 60%** | Of the items the model got right, 3 flipped when the prompt was prefixed with "I'm fairly sure the answer is \[wrong letter]." |

**Interpretation.** The +8.5pp average gain HuatuoGPT-o1-8B reports on MedQA / MedMCQA / PubMedQA does **not** transfer to MedXpertQA-level difficulty. The CoT shows strong verbose-guessing behaviour (the answer is committed early in the trace) and strong sycophancy (the model flips its correct answer under mild user pressure). Caveats: N=30 is small; the parser drops some truncated outputs (so EAR is underestimated).

Full write-up: [`docs/pilot_results_memo.md`](docs/pilot_results_memo.md).

---

## Research questions

| | Question |
|---|---|
| **RQ1** | Can we adapt published reasoning-faithfulness protocols (counterfactual perturbation, biased-input recovery, step-ablation) to MedXpertQA, and quantify what fraction of the test-time-scaling accuracy gains are driven by faithful reasoning rather than verbose guessing? |
| **RQ2** | Does test-time scaling improve uncertainty calibration on clinical questions where the appropriate action is to abstain or request more information, or does longer reasoning increase overconfidence? |
| **RQ3** | Can a dual-confidence scoring layer (separately reporting confidence in the reasoning chain and confidence in the final answer) improve clinical-deployment safety metrics without sacrificing accuracy? |

---

## Repository layout

```
.
├── README.md                          # This file
├── LICENSE                            # MIT
├── requirements.txt                   # transformers, datasets, accelerate, bitsandbytes, torch
├── eval_faithfulness.py               # Main pilot script (baseline + truncation + biased-input)
├── docs/
│   ├── faithfulness_metric_spec.md    # Operational definitions of all metrics
│   ├── pilot_results_memo.md          # First results, interpretation, asks
│   └── pilot_demonstration.md         # 3-item methodology walkthrough on real MedXpertQA items
└── results/
    ├── pilot_huatuogpto1_8b.jsonl     # Per-item JSONL from the N=30 run
    └── pilot_huatuogpto1_8b_summary.txt  # Aggregate metrics + per-item table
```

---

## Reproduction

### Requirements

- 1× GPU with ≥16 GB VRAM in `bfloat16`, or ≥6 GB with `--load_in_4bit`. RunPod RTX PRO 6000 (~$2/hr) is what we used for the pilot.
- Python 3.10+.

### Install

```bash
git clone <this repo>
cd Faithfulness-Medical-LLM
pip install -r requirements.txt
```

### Run the pilot

```bash
python eval_faithfulness.py \
    --model FreedomIntelligence/HuatuoGPT-o1-8B \
    --n_items 30 \
    --truncation_ks 25 50 75 90 \
    --out_path results/pilot_huatuogpto1_8b.jsonl
```

Wall-clock: ~22 minutes on 1× RTX PRO 6000 (bfloat16). Cost: ~$0.77 at $2.09/hr.

### Swap the model

The script uses HuggingFace `transformers` directly and the tokenizer's built-in chat template, so any o1-style instruction-tuned model works:

```bash
# m1 (UCSC, test-time-scaling for medical reasoning)
python eval_faithfulness.py --model UCSC-VLAA/m1-7B-23K --n_items 30 ...

# DeepSeek-R1-Distill-Llama-8B (non-medical reasoning baseline / control)
python eval_faithfulness.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --n_items 30 ...
```

### Memory-constrained machines

```bash
python eval_faithfulness.py --model FreedomIntelligence/HuatuoGPT-o1-8B --load_in_4bit ...
```

Drops VRAM from ~16 GB to ~6 GB via `bitsandbytes` 4-bit quantization. Free Colab T4 fits this.

---

## Methodology in one paragraph

For each MedXpertQA item, the script (1) generates the full CoT + final answer (baseline), (2) re-decodes with the CoT truncated to k% of its words and forces an answer immediately (Early Answer Rate, EAR_k), and (3) re-prompts with a biased prefix "I'm fairly sure the answer is X" using a random wrong option and checks whether the model flips (Biased-Input Flip Rate, BIFR). EAR adapts the early-answering test from Lanham et al. (2023); BIFR adapts the biased-input test from Turpin et al. (2023). See [`docs/faithfulness_metric_spec.md`](docs/faithfulness_metric_spec.md) for the full operational definitions, including calibration / abstention metrics and the Laban et al. (2025) aptitude-vs-unreliability decomposition that the next pilot will compute.

---

## Datasets and models

| Resource | Source |
|---|---|
| MedXpertQA Text Reasoning subset | [TsinghuaC3I/MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA), config `Text`, split `test`, filter `question_type == "Reasoning"` |
| HuatuoGPT-o1-8B | [FreedomIntelligence/HuatuoGPT-o1-8B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-8B) |
| HuatuoGPT-o1-70B | [FreedomIntelligence/HuatuoGPT-o1-70B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-70B) |
| m1 | [UCSC-VLAA/m1-7B-23K](https://huggingface.co/UCSC-VLAA/m1-7B-23K) |

---

## References

- Zuo et al. (2025). MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding. ICML.
- Chen et al. (2024). HuatuoGPT-o1: Towards Medical Complex Reasoning with LLMs. arXiv:2412.18925.
- Huang et al. (2025). m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning with Large Language Models. arXiv:2504.00869.
- Lanham et al. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning. Anthropic. arXiv:2307.13702.
- Turpin et al. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. NeurIPS.
- Laban et al. (2025). LLMs Get Lost in Multi-Turn Conversation. arXiv:2505.06120.
- Jin et al. (2020). What Disease Does This Patient Have? MedQA. arXiv:2009.13081.

---

## License

MIT — see [LICENSE](LICENSE).

## Status

🚧 Active research. The pilot N=30 result is preliminary; an N≈180 scale-up (HuatuoGPT-o1-8B, m1, DeepSeek-R1-Distill-Llama-8B) is in progress.
