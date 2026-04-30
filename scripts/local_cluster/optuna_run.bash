#!/bin/bash
#SBATCH --job-name=nspb_tune_synthetic
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G

module load pytorch
module load openmpi/4.1.6
module load cuda/11.8

echo "Starting hyperparameter tuning job..."

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TMPDIR="${REPO_ROOT}/tmp"
mkdir -p "${TMPDIR}"

RANDOM_SEED=3
N_TRIALS=70000
EVAL_WARMUP_STEPS=10
MODEL_KEY="KF"
TRAIN_DIR="${REPO_ROOT}/data/synthetic/theta_driven/train"
TEST_DIR="${REPO_ROOT}/data/synthetic/theta_driven/test"

srun --mpi=pmix -n "${SLURM_NTASKS}" python -u "${REPO_ROOT}/scripts/tune_synthetic.py" \
  --train-dir "$TRAIN_DIR" \
  --test-dir "$TEST_DIR" \
  --model "$MODEL_KEY" \
  --n-trials "$N_TRIALS" \
  --seed "$RANDOM_SEED" \
  --warmup "$EVAL_WARMUP_STEPS" \
  --quiet-optuna
