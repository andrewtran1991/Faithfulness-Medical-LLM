#!/usr/bin/env bash
# Re-run the N=30 pilot with the truncation parsing fix.
# Same seed (42) => same 30 items as the original pilot for direct comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-FreedomIntelligence/HuatuoGPT-o1-8B}"
EXTRA_ARGS=()
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load_in_4bit)
fi

echo "=== Re-running N=30 pilot: ${MODEL} ==="
python eval_faithfulness.py \
  --model "$MODEL" \
  --n_items 30 \
  --seed 42 \
  --truncation_ks 25 50 75 90 \
  --truncation_max_new_tokens 64 \
  --out_path results/pilot_huatuogpto1_8b_rerun.jsonl \
  --summary_path results/pilot_huatuogpto1_8b_rerun_summary.txt \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Compare against the original:"
echo "  results/pilot_huatuogpto1_8b_summary.txt"
echo "  results/pilot_huatuogpto1_8b_rerun_summary.txt"
