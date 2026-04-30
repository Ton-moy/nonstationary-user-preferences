#!/bin/bash
#SBATCH --job-name=nspb_kl_wasserstein
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load pytorch
module load cuda/11.8

echo "Starting KL + Wasserstein evaluation job..."
echo "Host: $(hostname)"
echo "Date: $(date)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TMPDIR="${REPO_ROOT}/tmp"
mkdir -p "${TMPDIR}"

TRAIN_DIR="${REPO_ROOT}/data/synthetic/theta_driven/train"
TEST_DIR="${REPO_ROOT}/data/synthetic/theta_driven/test"

WARMUP_STEPS=10
PD_THRESHOLDS="0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.6"
RELP_THRESHOLDS="0.25"
CAP_LAG=10
MODELS="KF BLR vbBLR fBLR AROW BLRsw PBLR"

PARAMS_JSON="${REPO_ROOT}/configs/model_hyperparameters_synthetic.json"
EVAL_SCRIPT="${REPO_ROOT}/scripts/evaluate_synthetic_distances.py"

echo "Train dir      : ${TRAIN_DIR}"
echo "Test dir       : ${TEST_DIR}"
echo "Warmup steps   : ${WARMUP_STEPS}"
echo "PD thresholds  : ${PD_THRESHOLDS}"
echo "RelP thresholds: ${RELP_THRESHOLDS}"
echo "Cap lag        : ${CAP_LAG}"
echo "Models         : ${MODELS}"
echo "Params         : ${PARAMS_JSON}"
echo "Script         : ${EVAL_SCRIPT}"
echo

python "${EVAL_SCRIPT}" \
  --models ${MODELS} \
  --train-data-dir "${TRAIN_DIR}" \
  --test-data-dir  "${TEST_DIR}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --pd-thresholds "${PD_THRESHOLDS}" \
  --relp-thresholds "${RELP_THRESHOLDS}" \
  --cap-lag "${CAP_LAG}" \
  --params-by-model-json "${PARAMS_JSON}" \
  --relp-mode l1_ratio

echo
echo "KL + Wasserstein evaluation job completed!"
echo "Date: $(date)"
