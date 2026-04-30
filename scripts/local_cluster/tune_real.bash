#!/bin/bash
#SBATCH --job-name=nspb_tune_real
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G

module load pytorch
module load openmpi/4.1.6
module load cuda/11.8

echo "Starting real-data hyperparameter tuning job..."

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TMPDIR="${REPO_ROOT}/tmp"
mkdir -p "${TMPDIR}"

MODEL_KEY="KF-AF"
N_TRIALS=30000
RANDOM_SEED=3
MIN_RATING_DIFF=2.0
COS_THRESHOLD="cos090"   # cos090 or cos095

TRAIN_DIR="${REPO_ROOT}/data/real/${COS_THRESHOLD}/train"
TEST_DIR="${REPO_ROOT}/data/real/${COS_THRESHOLD}/test"
BOOK_TOPICS_NPZ="${REPO_ROOT}/data/real/book_topics.npz"
OUTPUT_DIR="${REPO_ROOT}/results/real/tune/${MODEL_KEY}"   # e.g. results/real/tune/KF-AF

echo "Model          : ${MODEL_KEY}"
echo "Cosine dir     : ${COS_THRESHOLD}"
echo "Trials         : ${N_TRIALS}"
echo "Min rating diff: ${MIN_RATING_DIFF}"
echo "Train dir      : ${TRAIN_DIR}"
echo "Test dir       : ${TEST_DIR}"
echo "Output dir     : ${OUTPUT_DIR}"
echo

srun --mpi=pmix -n "${SLURM_NTASKS}" python -u "${REPO_ROOT}/scripts/tune_real.py" \
  --train-dir "${TRAIN_DIR}" \
  --test-dir "${TEST_DIR}" \
  --book-topics-npz "${BOOK_TOPICS_NPZ}" \
  --model "${MODEL_KEY}" \
  --n-trials "${N_TRIALS}" \
  --seed "${RANDOM_SEED}" \
  --min-rating-diff "${MIN_RATING_DIFF}" \
  --output-dir "${OUTPUT_DIR}"

echo
echo "Real-data tuning job completed!"
echo "Date: $(date)"
