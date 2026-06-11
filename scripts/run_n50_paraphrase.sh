#!/usr/bin/env bash
# Paraphrase stability (PS) on N=50 for HuatuoGPT-o1-8B and m1-7B (seed=42).
# Reuses cached N=200 fixed runs; only calls Claude + short GPU decode per item.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

nvidia-smi -L || { echo "ERROR: no GPU visible" >&2; exit 1; }

EXTRA_ARGS=()
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load_in_4bit)
fi

run_model() {
  local model="$1"
  local slug="$2"
  local prior="$3"
  echo ""
  echo "========================================"
  echo "PS run: ${model} (N=50, paraphrase-only)"
  echo "Cache:  ${prior}"
  echo "========================================"
  python eval_faithfulness.py \
    --model "$model" \
    --n_items 50 \
    --seed 42 \
    --truncation_ks 25 50 75 90 \
    --truncation_max_new_tokens 16 \
    --enable_paraphrase \
    --paraphrase_only_from "$prior" \
    --out_path "results/n50_${slug}_paraphrase.jsonl" \
    --summary_path "results/n50_${slug}_paraphrase_summary.txt" \
    "${EXTRA_ARGS[@]}"
}

run_model "FreedomIntelligence/HuatuoGPT-o1-8B" "huatuogpt-o1-8b" \
  "results/n200_huatuogpt-o1-8b_fixed.jsonl"

run_model "UCSC-VLAA/m1-7B-23K" "m1-7b-23k" \
  "results/n200_m1-7b-23k_fixed.jsonl"

echo ""
echo "Done. Summaries:"
echo "  results/n50_huatuogpt-o1-8b_paraphrase_summary.txt"
echo "  results/n50_m1-7b-23k_paraphrase_summary.txt"
