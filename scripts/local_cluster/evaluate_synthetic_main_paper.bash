#!/bin/bash
# Reproduces main-paper Table 3: p-hat-driven mixed-selection results for PS and PB.
# Usage: sbatch scripts/local_cluster/evaluate_synthetic_main_paper.bash [model ...]
# If no models are given, all 8 models are run in paper order.

#SBATCH --job-name=nspb_table3_main
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -eo pipefail

module load pytorch/2.3.0-cuda12.1 || module load pytorch

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# --- Run settings (edit here) ---
DATA_DIR="${REPO_ROOT}/data/synthetic/phat_driven_mixed"
PARAMS_JSON="${REPO_ROOT}/configs/model_hyperparameters_synthetic.json"
WARMUP_STEPS=10
RELPE_THRESHOLD=0.25   # paper uses 0.25
RELPE_MODE="l1_ratio"  # paper uses l1_ratio
CAP_LAG=10
OUT_DIR="${REPO_ROOT}/results/main"

# Accept optional model list; default to all 8 models
MODELS=("${@:-all}")

# --- Run ---
mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

python -u "${REPO_ROOT}/scripts/evaluate_synthetic_main.py" \
  --data-dir        "${DATA_DIR}" \
  --models          "${MODELS[@]}" \
  --warmup-steps    "${WARMUP_STEPS}" \
  --relpe-threshold "${RELPE_THRESHOLD}" \
  --relpe-mode      "${RELPE_MODE}" \
  --cap-lag         "${CAP_LAG}" \
  --params-by-model-json "${PARAMS_JSON}" \
  --output-txt      "${OUT_DIR}/synthetic_main_table3_${STAMP}.txt" \
  --output-csv      "${OUT_DIR}/synthetic_main_table3_${STAMP}.csv"
