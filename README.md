# A Benchmark Suite for Online Learning of Non-Stationary User Preferences

This repository contains benchmark artifacts for evaluating online learners on non-stationary user preference tracking. It includes two complementary benchmarks:

- **Synthetic benchmark** — controlled scenarios with ground-truth preference vectors and preference-change labels
- **Real benchmark** — Goodreads book review data with paired items matched by topic similarity

## Models

Eight online Bayesian learners are evaluated (paper names used throughout):

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

## Synthetic Benchmark

The synthetic benchmark covers six user-behavior scenarios, three recommendation settings, and two model-driven selection policies. It reproduces the paper's reported Table 1 totals:

| Setting | Users | Items | Interactions | p-change events |
|---|---:|---:|---:|---:|
| theta-driven | 90 | 5,996 | 5,996 | 411 |
| p-driven | 90 | 5,996 | 5,996 | 411 |
| p-hat-driven top-2% | 180 | 2,960 | 11,992 | 812 |
| p-hat-driven mixed | 180 | 2,960 | 11,992 | 812 |
| **Total** | **540** | **17,912** | **35,976** | **2,436** |

### Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make smoke
```

Full generation pipeline:

```bash
# 1. Generate item catalog (2,960 items)
python scripts/01_generate_synthetic.py --config configs/synthetic.yaml --catalog

# 2. Generate theta-driven and p-driven base logs (180 users)
python scripts/01_generate_synthetic.py --config configs/synthetic.yaml --all

# 3. Generate model-driven (p-hat) logs
python scripts/02_generate_phat.py --input-dir data/synthetic/theta_driven \
    --catalog-dir data/synthetic/phat_item_catalog \
    --output-dir data/synthetic/phat_driven_mixed --model KF-AF
```

## Real Benchmark

Uses Goodreads book reviews. Per-user JSON files are not distributed; place them under `data/real/` following the layout in [data/real/README.md](data/real/README.md).

```bash
# Tune one model (MPI, 30 000 trials)
mpirun -n 16 python scripts/tune_real.py \
    --train-dir data/real/cos090/train \
    --test-dir  data/real/cos090/test \
    --book-topics-npz data/real/book_topics.npz \
    --model KF-AF --n-trials 30000 --min-rating-diff 2.0 \
    --output-dir results/real/tune/KF-AF

# Evaluate with tuned hyperparameters
mpirun -n 16 python scripts/evaluate_real.py \
    --train-dir data/real/cos090/train \
    --test-dir  data/real/cos090/test \
    --book-topics-npz data/real/book_topics.npz \
    --params-by-model-json configs/model_hyperparameters_real.json \
    --output-dir results/real/eval/cos090
```

## Repository Structure

```text
configs/
  synthetic.yaml                  Canonical parameters for synthetic benchmark
  model_hyperparameters_synthetic.json
                                  Tuned hyperparameters for synthetic data
  model_hyperparameters_real.json Tuned hyperparameters for real data

data/
  synthetic/                      Generated synthetic files (git-ignored except READMEs)
  real/                           Real Goodreads data (git-ignored; see data/real/README.md)
  samples/                        Tiny committed examples for smoke tests

docs/
  SYNTHETIC_DATASET.md            Synthetic benchmark organization guide

scripts/
  01_generate_synthetic.py        Generate item catalog and synthetic user logs
  02_generate_phat.py             Generate model-driven p-hat datasets
  tune_synthetic.py               Optuna tuning for theta-driven and p-driven data
  tune_phat.py                    Optuna tuning for p-hat/self-directed data
  tune_real.py                    MPI+Optuna tuning for real Goodreads data
  evaluate_synthetic_main.py      Main-paper Table 3 evaluator (p-hat mixed PS/PB)
  evaluate_synthetic.py           Appendix evaluator for theta-driven and p-driven data
  evaluate_synthetic_distances.py Appendix KL/Wasserstein comparison evaluator
  evaluate_phat.py                Appendix evaluator for p-hat/self-directed data
  evaluate_real.py                MPI evaluation for real Goodreads data

src/nspb/
  scenarios.py                    Scenario definitions and paper-name mappings
  catalog.py                      p-hat item catalog construction
  models.py                       Online Bayesian learner implementations
  posterior_distances.py          KL and Wasserstein posterior distance utilities

tests/                            Minimal quality gates

scripts/local_cluster/            SLURM job scripts (git-ignored)
```

## Anonymization Note

This repository is intended for anonymous submission. Do not commit `.git/`, SLURM logs, absolute local paths, raw Goodreads records, or personal/institutional identifiers.
