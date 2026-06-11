#!/usr/bin/env bash
# Re-run truncation only for m1-7B N=200 with letter-constrained decoding.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PRIOR="${PRIOR:-results/n200_m1-7b-23k.jsonl}"
OUT="${OUT:-results/n200_m1-7b-23k_fixed.jsonl}"
SUMMARY="${SUMMARY:-results/n200_m1-7b-23k_fixed_summary.txt}"
BEFORE="${BEFORE:-results/n200_m1-7b-23k_summary.txt}"

EXTRA_ARGS=()
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load_in_4bit)
fi

[[ -f "$PRIOR" ]] || { echo "ERROR: prior not found: $PRIOR" >&2; exit 1; }

echo "=== m1-7B N=200 truncation re-run (parsing fix) ==="
echo "Prior:  $PRIOR"
echo "Output: $OUT"
echo ""

python eval_faithfulness.py \
  --model UCSC-VLAA/m1-7B-23K \
  --n_items 200 \
  --seed 42 \
  --truncation_ks 25 50 75 90 \
  --truncation_max_new_tokens 16 \
  --rerun_truncation_from "$PRIOR" \
  --out_path "$OUT" \
  --summary_path "$SUMMARY" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Compare empty-parse rates:"
python scripts/compare_summaries.py "$BEFORE" "$SUMMARY"
