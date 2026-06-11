#!/usr/bin/env bash
# Scale-up: N=180 items on three models (same seed => same item set).
# At ~17% accuracy, 180 items yields ~30 originally-correct for a defensible BIFR.
#
# Usage:
#   ./scripts/scale_up_n180.sh              # all three models
#   MODEL=UCSC-VLAA/m1-7B-23K ./scripts/scale_up_n180.sh   # single model
#   LOAD_IN_4BIT=1 ./scripts/scale_up_n180.sh               # 4-bit quant
#   ENABLE_PARAPHRASE=1 ./scripts/scale_up_n180.sh   # reads ANTHROPIC_API_KEY from .env
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N_ITEMS="${N_ITEMS:-180}"
SEED="${SEED:-42}"
EXTRA_ARGS=()
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load_in_4bit)
fi
if [[ "${ENABLE_PARAPHRASE:-0}" == "1" ]]; then
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
  PROVIDER="${PARAPHRASE_PROVIDER:-claude}"
  if [[ "$PROVIDER" == "claude" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ENABLE_PARAPHRASE=1 requires ANTHROPIC_API_KEY in .env" >&2
    exit 1
  fi
  if [[ "$PROVIDER" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: ENABLE_PARAPHRASE=1 with PARAPHRASE_PROVIDER=openai requires OPENAI_API_KEY in .env" >&2
    exit 1
  fi
  EXTRA_ARGS+=(--enable_paraphrase --paraphrase_provider "$PROVIDER")
fi

run_model() {
  local model="$1"
  local slug
  slug="$(echo "$model" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '.' '_')"
  local out="results/n${N_ITEMS}_${slug}.jsonl"
  local summary="results/n${N_ITEMS}_${slug}_summary.txt"

  echo ""
  echo "========================================"
  echo "Model: ${model}"
  echo "Output: ${out}"
  echo "========================================"

  python eval_faithfulness.py \
    --model "$model" \
    --n_items "$N_ITEMS" \
    --seed "$SEED" \
    --truncation_ks 25 50 75 90 \
    --truncation_max_new_tokens 64 \
    --out_path "$out" \
    --summary_path "$summary" \
    "${EXTRA_ARGS[@]}"
}

if [[ -n "${MODEL:-}" ]]; then
  run_model "$MODEL"
else
  run_model "FreedomIntelligence/HuatuoGPT-o1-8B"
  run_model "UCSC-VLAA/m1-7B-23K"
  run_model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
fi

echo ""
echo "All runs complete. Summaries in results/n${N_ITEMS}_*_summary.txt"
