#!/bin/bash
# Reproduces Appendix Tables 2 and 3 of the synthetic benchmark.
#   Table 2: all 3 settings (θ-driven, p-driven, p̂-driven top-2%), aggregated over 6 scenarios.
#   Table 3: p̂-driven mixed selection, per scenario: PS, PSC1T, PB, PBC1T.
# Usage: sbatch scripts/local_cluster/evaluate_synthetic_appendix.bash

#SBATCH --job-name=nspb_appendix
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -eo pipefail

module load pytorch/2.3.0-cuda12.1 || module load pytorch

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# --- Run settings (edit here) ---
THETA_DIR="${REPO_ROOT}/data/synthetic/theta_driven/test"
P_DIR="${REPO_ROOT}/data/synthetic/p_driven/test"
PHAT_TOP2_DIR="${REPO_ROOT}/data/synthetic/phat_driven_top2"
PHAT_MIXED_DIR="${REPO_ROOT}/data/synthetic/phat_driven_mixed"
PARAMS_JSON="${REPO_ROOT}/configs/model_hyperparameters_synthetic.json"
WARMUP_STEPS=10
RELPE_THRESHOLD=0.25   # paper uses 0.25
RELPE_MODE="l1_ratio"  # paper uses l1_ratio
CAP_LAG=10
OUT_DIR="${REPO_ROOT}/results/appendix"

# --- Run ---
mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

python -u "${REPO_ROOT}/scripts/evaluate_synthetic_appendix.py" \
  --theta-dir           "${THETA_DIR}" \
  --p-dir               "${P_DIR}" \
  --phat-top2-dir       "${PHAT_TOP2_DIR}" \
  --phat-mixed-dir      "${PHAT_MIXED_DIR}" \
  --warmup-steps        "${WARMUP_STEPS}" \
  --relpe-threshold     "${RELPE_THRESHOLD}" \
  --relpe-mode          "${RELPE_MODE}" \
  --cap-lag             "${CAP_LAG}" \
  --params-by-model-json "${PARAMS_JSON}" \
  --output-dir          "${OUT_DIR}"
