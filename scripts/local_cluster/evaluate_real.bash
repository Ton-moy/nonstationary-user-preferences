#!/bin/bash
#SBATCH --job-name=nspb_eval_real
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G

module load pytorch
module load openmpi/4.1.6
module load cuda/11.8

echo "Starting real-data evaluation job..."
echo "Date: $(date)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TMPDIR="${REPO_ROOT}/tmp"
mkdir -p "${TMPDIR}"

COS_THRESHOLD="cos090"   # cos090 or cos095
TRAIN_DIR="${REPO_ROOT}/data/real/${COS_THRESHOLD}/train"
TEST_DIR="${REPO_ROOT}/data/real/${COS_THRESHOLD}/test"
BOOK_TOPICS_NPZ="${REPO_ROOT}/data/real/book_topics.npz"
PARAMS_JSON="${REPO_ROOT}/configs/model_hyperparameters_real.json"
OUTPUT_DIR="${REPO_ROOT}/results/real/eval/${COS_THRESHOLD}"

MIN_RATING_DIFF=2.0
WARMUP=10
MIN_COSINES="0.90,0.95"
MAX_PAIR_GAP_DAYS="all,7,14"
MODELS="KF-AF AROW BLR BLR-VB BLR-FF BLR-SW BLR-PP BLR-NIG"

echo "Cosine dir     : ${COS_THRESHOLD}"
echo "Train dir      : ${TRAIN_DIR}"
echo "Test dir       : ${TEST_DIR}"
echo "Params JSON    : ${PARAMS_JSON}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "Min rating diff: ${MIN_RATING_DIFF}"
echo "Warmup         : ${WARMUP}"
echo "Min cosines    : ${MIN_COSINES}"
echo "Max gap days   : ${MAX_PAIR_GAP_DAYS}"
echo "Models         : ${MODELS}"
echo

srun --mpi=pmix -n "${SLURM_NTASKS}" python -u "${REPO_ROOT}/scripts/evaluate_real.py" \
  --train-dir "${TRAIN_DIR}" \
  --test-dir "${TEST_DIR}" \
  --book-topics-npz "${BOOK_TOPICS_NPZ}" \
  --params-by-model-json "${PARAMS_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --min-rating-diff "${MIN_RATING_DIFF}" \
  --warmup "${WARMUP}" \
  --min-cosines "${MIN_COSINES}" \
  --max-pair-gap-days "${MAX_PAIR_GAP_DAYS}" \
  --models "${MODELS}"

echo
echo "Real-data evaluation job completed!"
echo "Date: $(date)"
