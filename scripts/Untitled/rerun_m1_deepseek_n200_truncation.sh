#!/usr/bin/env bash
# Re-run truncation for m1-7B then DeepSeek-R1-Distill (N=200 each, parsing fix).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "========== 1/2 m1-7B =========="
"$ROOT/scripts/rerun_m1_n200_truncation.sh"

echo ""
echo "========== 2/2 DeepSeek-R1-Distill =========="
"$ROOT/scripts/rerun_deepseek_n200_truncation.sh"

echo ""
echo "All truncation re-runs complete."
