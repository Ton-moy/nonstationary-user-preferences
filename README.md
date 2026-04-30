# A Benchmark Suite for Online Learning of Non-Stationary User Preferences

This repository accompanies the paper *"A Benchmark Suite for Online Learning of Non-Stationary User Preferences in Recommender Systems"*. It provides reproducible code, configurations, and evaluation scripts for benchmarking online learning algorithms on the task of tracking user preferences when those preferences change over time.

The benchmark has two complementary components:

- **Synthetic benchmark.** Controlled, parameterized user-behavior scenarios with ground-truth preference vectors and preference-change event labels. Enables direct measurement of preference tracking, not just rating prediction.
- **Real benchmark.** A pipeline built on Goodreads book reviews, in which preference change is signaled by rating discrepancies between near-duplicate book pairs read in close temporal succession.

---

## What This Benchmark Evaluates

Conventional recommender system benchmarks emphasize aggregate rating prediction (e.g., RMSE, MSE), which can be insensitive to whether a model actually adapts its internal preference estimate when a user's preferences change. This benchmark is designed to expose that gap. It evaluates each model on three complementary axes:

- **Preference change detection** (ROC-AUC and PR-AUC over change events vs. stable steps)
- **Preference tracking** (tracking ratio, mean tracking lag, and capped MTL)
- **Rating prediction** (one-step-ahead prequential MSE)

Models that excel on rating prediction can fail on tracking, and vice versa: this benchmark is constructed to make those dissociations visible.

---

## Models Evaluated

Eight online Bayesian learners are included. Paper names are used consistently across the codebase and configuration files.

| Paper name | Description |
|---|---|
| KF-AF | Kalman filter with adaptive forgetting |
| AROW | Adaptive Regularization of Weights |
| BLR | Bayesian linear regression |
| BLR-VB | BLR with variance bound |
| BLR-FF | BLR with forgetting factor |
| BLR-SW | BLR with sliding window |
| BLR-PP | BLR with power prior |
| BLR-NIG | BLR with Normal-Inverse-Gamma prior |

Implementations are in `src/nspb/models.py`. Each model exposes a common online interface so that benchmark scripts can swap models without scenario-specific code.

---

## Synthetic Benchmark

The synthetic benchmark covers six user-behavior scenarios, three recommendation settings, and two model-driven selection policies. The dataset totals reproduce those reported in the paper:

| Setting | Users | Items | Interactions | Preference-change events |
|---|---:|---:|---:|---:|
| theta-driven | 90 | 5,996 | 5,996 | 411 |
| p-driven | 90 | 5,996 | 5,996 | 411 |
| p-hat-driven (top-2%) | 180 | 2,960 | 11,992 | 812 |
| p-hat-driven (mixed) | 180 | 2,960 | 11,992 | 812 |
| **Total** | **540** | **17,912** | **35,976** | **2,436** |

### Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make smoke
```

`make smoke` runs the full pipeline on a small committed sample (`data/samples/`) and verifies that all stages execute end-to-end. Expect the smoke test to complete in under a minute.

### Full generation pipeline

```bash
# 1. Generate item catalog (2,960 items)
python scripts/01_generate_synthetic.py \
    --config configs/synthetic.yaml \
    --catalog

# 2. Generate theta-driven and p-driven base logs (180 users)
python scripts/01_generate_synthetic.py \
    --config configs/synthetic.yaml \
    --all

# 3. Generate p-hat-driven logs by replaying theta-driven sequences
#    through a chosen model's selection policy (top-2% or mixed).
python scripts/02_generate_phat.py \
    --input-dir data/synthetic/theta_driven \
    --catalog-dir data/synthetic/phat_item_catalog \
    --output-dir data/synthetic/phat_driven_mixed \
    --model KF-AF
```

### Evaluation

```bash
# Main-paper Table 3 (p-hat-driven mixed, PS and PB scenarios)
python scripts/evaluate_synthetic_main.py \
    --data-dir data/synthetic/phat_driven_mixed \
    --params-by-model-json configs/model_hyperparameters_synthetic.json \
    --output-dir results/synthetic/main

# Full appendix tables (theta-driven and p-driven)
python scripts/evaluate_synthetic.py \
    --data-dir data/synthetic/theta_driven \
    --params-by-model-json configs/model_hyperparameters_synthetic.json \
    --output-dir results/synthetic/appendix
```

For details on scenario definitions, item-catalog construction, and the four-layer organization of the synthetic dataset, see [`docs/SYNTHETIC_DATASET.md`](docs/SYNTHETIC_DATASET.md).

---

## Real Benchmark

The real benchmark builds on Goodreads book review data. Per-user JSON files are not redistributed in this repository for licensing reasons; users must obtain the source dataset separately. See [`data/real/README.md`](data/real/README.md) for the expected directory layout, file format, and a description of the preprocessing pipeline.

The benchmark uses preprocessed pair lists where each book is matched against later books read by the same user that share high topic similarity (cosine threshold) and lie within a configurable time window (default 7 days or 14 days). Rating discrepancies between paired items serve as a proxy preference-change signal.

### Tuning model hyperparameters

```bash
# Tune one model (MPI; 30,000 Optuna trials)
mpirun -n 16 python scripts/tune_real.py \
    --train-dir data/real/cos090/train \
    --test-dir  data/real/cos090/test \
    --book-topics-npz data/real/book_topics.npz \
    --model KF-AF \
    --n-trials 30000 \
    --min-rating-diff 2.0 \
    --output-dir results/real/tune/KF-AF
```

### Evaluation

```bash
mpirun -n 16 python scripts/evaluate_real.py \
    --train-dir data/real/cos090/train \
    --test-dir  data/real/cos090/test \
    --book-topics-npz data/real/book_topics.npz \
    --params-by-model-json configs/model_hyperparameters_real.json \
    --output-dir results/real/eval/cos090
```

The evaluation script writes both an aggregate summary (`GLOBAL__all_models_summary.json`) and per-user metric CSVs that downstream paired statistical tests can consume.

---

## Reproducing Paper Results

The configuration files committed in `configs/` contain the exact tuned hyperparameters used to produce the tables in the paper. To reproduce:

```bash
# Synthetic Table 3 (main paper)
python scripts/evaluate_synthetic_main.py \
    --data-dir data/synthetic/phat_driven_mixed \
    --params-by-model-json configs/model_hyperparameters_synthetic.json \
    --output-dir results/synthetic/table3

# Real-data Table 4 (main paper)
mpirun -n 16 python scripts/evaluate_real.py \
    --train-dir data/real/cos090/train \
    --test-dir  data/real/cos090/test \
    --book-topics-npz data/real/book_topics.npz \
    --params-by-model-json configs/model_hyperparameters_real.json \
    --output-dir results/real/table4
```

Synthetic results are deterministic when the random seeds in `configs/synthetic.yaml` are preserved. Real-data results depend on the Goodreads dataset version used; the version expected by this code is documented in `data/real/README.md`.

---

## Repository Structure

```text
configs/
  synthetic.yaml                          Canonical parameters for synthetic benchmark
  model_hyperparameters_synthetic.json    Tuned hyperparameters for synthetic data
  model_hyperparameters_real.json         Tuned hyperparameters for real data

data/
  synthetic/                              Generated synthetic files (git-ignored except READMEs)
  real/                                   Real Goodreads data (git-ignored; see data/real/README.md)
  samples/                                Tiny committed examples for smoke tests

docs/
  SYNTHETIC_DATASET.md                    Synthetic benchmark organization guide

scripts/
  01_generate_synthetic.py                Generate item catalog and synthetic user logs
  02_generate_phat.py                     Generate model-driven p-hat datasets
  tune_synthetic.py                       Optuna tuning for theta-driven and p-driven data
  tune_phat.py                            Optuna tuning for p-hat/self-directed data
  tune_real.py                            MPI + Optuna tuning for real Goodreads data
  evaluate_synthetic_main.py              Main-paper Table 3 evaluator (p-hat mixed, PS/PB)
  evaluate_synthetic.py                   Appendix evaluator for theta-driven and p-driven data
  evaluate_synthetic_distances.py         Appendix KL / Wasserstein comparison evaluator
  evaluate_phat.py                        Appendix evaluator for p-hat / self-directed data
  evaluate_real.py                        MPI evaluation for real Goodreads data

src/nspb/
  scenarios.py                            Scenario definitions and paper-name mappings
  catalog.py                              p-hat item catalog construction
  models.py                               Online Bayesian learner implementations
  posterior_distances.py                  KL and Wasserstein posterior distance utilities

tests/                                    Minimal quality gates

scripts/local_cluster/                    SLURM job scripts (git-ignored)
```

---

## Requirements

- Python 3.10 or newer
- NumPy, SciPy, scikit-learn (installed via `pip install -e ".[dev]"`)
- For real-benchmark scripts: `mpi4py` and a working MPI installation (OpenMPI or MPICH)
- Optional: Optuna, for hyperparameter tuning

Full dependency list is in `pyproject.toml`.

---

