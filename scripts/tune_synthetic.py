#!/usr/bin/env python3
"""
MPI + Optuna (rating-only) tuning + evaluation.

Optuna objective:
  - TRAIN MSE (after-update): (r_t, p_hat_t^T x_t) where p_hat_t is AFTER update on (x_t, r_t)

Final report:
  - TRAIN MSE (after-update)
  - TEST  MSE (before-update-nextstep): (r_{t+1}, p_t^T x_{t+1}) where p_t is BEFORE update on (x_{t+1}, r_{t+1})

Notes:
- No hinge loss, no preference-change labels.
- NIW removed; includes NIG + PBLR.
- Designed to run with: srun python -u this_script.py ...
"""

import os
import glob
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from mpi4py import MPI

from nspb.models import (
    KalmanFilter,
    BayesianModel,
    VarianceBoundedBayesianModel,
    BayesianForgettingFactorModel,
    AROW_Regression,
    BayesianSlidingWindowModel,
    PowerPriorBayesianModel,         # PBLR
    NormalInverseGammaModel,         # NIG
)

MODEL_CHOICES = ["KF-AF", "AROW", "BLR", "BLR-VB", "BLR-FF", "BLR-SW", "BLR-PP", "BLR-NIG"]

PAPER_TO_KEY = {
    "KF-AF":   "KF",
    "BLR-VB":  "vbBLR",
    "BLR-FF":  "fBLR",
    "BLR-SW":  "BLRsw",
    "BLR-PP":  "PBLR",
    "BLR-NIG": "NIG",
}


# -------------------------
# Data utilities
# -------------------------
@dataclass
class UserData:
    user_id: str
    X: np.ndarray  # (T, d)
    y: np.ndarray  # (T,)


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def rating_to_scaled_minus2to2(r: float) -> float:
    if 0.5 <= r <= 5.5:
        return r - 3.0
    return r


def load_user_file(path: str, x_key: str = "topic_vector", y_key: str = "rating") -> UserData:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X_list = []
    y_list = []
    for rec in data:
        if x_key not in rec:
            continue
        r = _safe_float(rec.get(y_key))
        if r is None:
            continue
        X_list.append(np.asarray(rec[x_key], dtype=np.float64))
        y_list.append(r)

    uid = os.path.splitext(os.path.basename(path))[0]

    if not X_list:
        return UserData(uid, np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64))

    X = np.vstack([x.reshape(1, -1) for x in X_list])
    y = np.asarray(y_list, dtype=np.float64)
    return UserData(uid, X, y)


def list_user_paths(data_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*.json")))


def shard_list(items: List[str], rank: int, size: int) -> List[str]:
    return [items[i] for i in range(len(items)) if (i % size) == rank]


def load_users_sharded(data_dir: str, comm: MPI.Comm) -> List[UserData]:
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        paths = list_user_paths(data_dir)
        if not paths:
            raise SystemExit(f"No JSON files found in {data_dir}")
    else:
        paths = None

    paths = comm.bcast(paths, root=0)
    my_paths = shard_list(paths, rank, size)

    users: List[UserData] = []
    for p in my_paths:
        u = load_user_file(p)
        if u.X.size == 0 or u.y.size == 0:
            continue
        users.append(u)
    return users


# -------------------------
# Models
# -------------------------
def build_model(model_key: str, d: int, params: Dict[str, float]):
    if model_key == "KF":
        return KalmanFilter(
            d,
            params.get("variance_p", 3.0),
            params.get("variance", 0.3),
            params.get("delta", 0.015),
            params.get("eta", 1e-5),
        )
    if model_key == "BLR":
        return BayesianModel(d, params.get("default_variance", 50.0), params.get("noise_precision", 0.33))
    if model_key == "vbBLR":
        return VarianceBoundedBayesianModel(d, params.get("default_variance", 15.0),
                                            params.get("noise_precision", 0.33),
                                            params.get("tau_v", 1.0))
    if model_key == "fBLR":
        return BayesianForgettingFactorModel(d, params.get("default_variance", 50.0), params.get("noise_precision", 0.33))
    if model_key == "AROW":
        return AROW_Regression(d, params.get("lam1", 0.8), params.get("lam2", 1.0))
    if model_key == "BLRsw":
        return BayesianSlidingWindowModel(d, int(params.get("m", 20)),
                                          params.get("default_variance", 50.0),
                                          params.get("noise_precision", 0.33))
    if model_key == "PBLR":
        return PowerPriorBayesianModel(d, params.get("alpha", 0.9),
                                       params.get("default_variance", 50.0),
                                       params.get("noise_precision", 0.33))
    if model_key == "NIG":
        return NormalInverseGammaModel(
            d,
            default_variance=params.get("default_variance", 50.0),
            default_a=params.get("default_a", 2.0),
            default_b=params.get("default_b", 2.0),
        )
    raise ValueError(f"Unknown model key: {model_key}")


def _predict_scalar(model, x_vec: np.ndarray) -> float:
    y_pred = model.predict(x_vec)
    if isinstance(y_pred, (np.ndarray, list, tuple)):
        return float(np.asarray(y_pred).reshape(-1)[0])
    return float(y_pred)


def _update_model(model, model_key: str, x_vec: np.ndarray, y_true: float, params: Dict[str, float]):
    if model_key == "fBLR":
        return model.update(x_vec, y_true, rho=float(params.get("rho", 0.98)))
    return model.update(x_vec, y_true)


# -------------------------
# MSE definitions
# -------------------------
def train_sse_count_after_update(users: List[UserData], model_key: str, params: Dict[str, float], warmup: int) -> Tuple[float, int]:
    """
    TRAIN:
      predict AFTER update on (x_t, r_t): r_t vs p_hat_t^T x_t.
    """
    sse = 0.0
    n = 0
    for u in users:
        X, y = u.X, u.y
        if X.size == 0 or y.size == 0:
            continue
        d = X.shape[1]
        model = build_model(model_key, d, params)

        for t in range(X.shape[0]):
            y_true = rating_to_scaled_minus2to2(float(y[t]))

            # update first
            _update_model(model, model_key, X[t], y_true, params)

            if t < warmup:
                continue

            y_pred = _predict_scalar(model, X[t])

            if math.isfinite(y_pred) and math.isfinite(y_true):
                err = y_pred - y_true
                sse += float(err * err)
                n += 1
    return float(sse), int(n)


def test_sse_count_before_update_nextstep(users: List[UserData], model_key: str, params: Dict[str, float], warmup: int) -> Tuple[float, int]:
    """
    TEST:
      predict BEFORE update on (x_{t}, r_{t}) for t>=1:
      compare r_t vs p_{t-1}^T x_t  (matches your r_{t+1} vs p_t^T x_{t+1}).
    """
    sse = 0.0
    n = 0
    for u in users:
        X, y = u.X, u.y
        if X.size == 0 or y.size == 0 or X.shape[0] < 2:
            continue
        d = X.shape[1]
        model = build_model(model_key, d, params)

        # initialize state with t=0 update
        y0 = rating_to_scaled_minus2to2(float(y[0]))
        _update_model(model, model_key, X[0], y0, params)

        for t in range(1, X.shape[0]):
            y_true = rating_to_scaled_minus2to2(float(y[t]))

            # predict before update
            y_pred = _predict_scalar(model, X[t])

            if t >= warmup:
                if math.isfinite(y_pred) and math.isfinite(y_true):
                    err = y_pred - y_true
                    sse += float(err * err)
                    n += 1

            # update now
            _update_model(model, model_key, X[t], y_true, params)

    return float(sse), int(n)


# -------------------------
# Optuna param suggestion
# -------------------------
def suggest_params(trial: optuna.Trial, model_key: str) -> Dict[str, float]:
    if model_key == "KF":
        return {
            "variance_p": trial.suggest_float("variance_p", 1e-2, 5e+1, log=True),
            "variance": trial.suggest_float("variance", 1e-2, 5e+1, log=True),
            "delta": trial.suggest_float("delta", 1e-3, 0.3, log=True),
            "eta": trial.suggest_float("eta", 1e-7, 1e-1, log=True),
        }
    if model_key == "BLR":
        return {
            "default_variance": trial.suggest_float("default_variance", 1e-2, 1e+2, log=True),
            "noise_precision": trial.suggest_float("noise_precision", 1e-2, 3e+1, log=True),
        }
    if model_key == "vbBLR":
        return {
            "default_variance": trial.suggest_float("default_variance", 1e-2, 1e+2, log=True),
            "noise_precision": trial.suggest_float("noise_precision", 1e-2, 3e+1, log=True),
            "tau_v": trial.suggest_float("tau_v", 0.3, 1.0, log=True),
        }
    if model_key == "fBLR":
        return {
            "default_variance": trial.suggest_float("default_variance", 1e-2, 5e+1, log=True),
            "noise_precision": trial.suggest_float("noise_precision", 1e-1, 3e+1, log=True),
            "rho": trial.suggest_float("rho", 0.95, 0.999),
        }
    if model_key == "AROW":
        return {
            "lam1": trial.suggest_float("lam1", 1e-2, 1e+2, log=True),
            "lam2": trial.suggest_float("lam2", 1e-2, 1e+2, log=True),
        }
    if model_key == "BLRsw":
        return {
            "default_variance": trial.suggest_float("default_variance", 1e-2, 1e+2, log=True),
            "noise_precision": trial.suggest_float("noise_precision", 1e-2, 3e+1, log=True),
            "m": trial.suggest_int("m", 2, 50),
        }
    if model_key == "PBLR":
        return {
            "alpha": trial.suggest_float("alpha", 0.85, 1.0),
            "default_variance": trial.suggest_float("default_variance", 1e-2, 1e+2, log=True),
            "noise_precision": trial.suggest_float("noise_precision", 1e-2, 3e+1, log=True),
        }
    if model_key == "NIG":
        return {
            "default_variance": trial.suggest_float("default_variance", 1e-2, 1e+2, log=True),
            "default_a": trial.suggest_float("default_a", 1.01, 50.0, log=True),
            "default_b": trial.suggest_float("default_b", 1e-3, 50.0, log=True),
        }
    raise ValueError(f"Unknown model key: {model_key}")


# -------------------------
# MPI Trial runner
# -------------------------
def run_one_trial(comm: MPI.Comm, train_users: List[UserData], model_key: str, params: Dict[str, float], warmup: int) -> Tuple[float, int]:
    # local SSE/count
    sse_local, n_local = train_sse_count_after_update(train_users, model_key, params, warmup)

    # reduce to rank0
    sse_total = comm.reduce(sse_local, op=MPI.SUM, root=0)
    n_total = comm.reduce(n_local, op=MPI.SUM, root=0)

    if comm.Get_rank() == 0:
        return float(sse_total), int(n_total)
    return 0.0, 0


def main():
    parser = argparse.ArgumentParser(description="MPI Optuna tuning for rating MSE (NIG/PBLR supported, NIW removed).")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--quiet-optuna", action="store_true")
    args = parser.parse_args()
    model_key = PAPER_TO_KEY.get(args.model, args.model)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if args.quiet_optuna:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    np.random.seed(int(args.seed))

    # Load sharded users on each rank
    train_users = load_users_sharded(args.train_dir, comm)
    test_users = load_users_sharded(args.test_dir, comm)

    # Ensure global non-empty (rank0 checks)
    local_train_n = len(train_users)
    global_train_n = comm.allreduce(local_train_n, op=MPI.SUM)
    if rank == 0 and global_train_n == 0:
        raise SystemExit(f"No valid training users found in {args.train_dir}")

    # Optuna loop: rank0 owns the study; broadcasts params each trial
    if rank == 0:
        sampler = optuna.samplers.TPESampler(seed=int(args.seed))
        study = optuna.create_study(direction="minimize", sampler=sampler)
    else:
        study = None

    def eval_params(params: Dict[str, float]) -> float:
        sse_total, n_total = run_one_trial(comm, train_users, model_key, params, int(args.warmup))
        if rank != 0:
            return 0.0
        if n_total <= 0:
            return 1e18
        return float(sse_total / n_total)

    # manual ask/tell so all ranks can participate each trial
    for _ in range(int(args.n_trials)):
        if rank == 0:
            trial = study.ask()
            params = suggest_params(trial, model_key)
        else:
            params = None

        params = comm.bcast(params, root=0)

        val = eval_params(params)

        if rank == 0:
            study.tell(trial, val)

    # Best params broadcast
    if rank == 0:
        best_params = dict(study.best_params)
        best_train_mse = float(study.best_value)
    else:
        best_params = None
        best_train_mse = None

    best_params = comm.bcast(best_params, root=0)
    best_train_mse = comm.bcast(best_train_mse, root=0)

    # Final TEST evaluation with your TEST definition
    test_sse_local, test_n_local = test_sse_count_before_update_nextstep(
        test_users, model_key, best_params, int(args.warmup)
    )
    test_sse_total = comm.reduce(test_sse_local, op=MPI.SUM, root=0)
    test_n_total = comm.reduce(test_n_local, op=MPI.SUM, root=0)

    if rank == 0:
        test_mse = float(test_sse_total / test_n_total) if test_n_total > 0 else float("inf")
        print("=============================================================")
        print(f"Model:            {args.model} (key={model_key})")
        print(f"Seed:             {args.seed}")
        print(f"Warmup:           {args.warmup}")
        print(f"Number of trials: {args.n_trials}")
        print("-------------------------------------------------------------")
        print(f"Best TRAIN MSE (after-update):  {best_train_mse:.6f}")
        print(f"Best TEST  MSE (before-update): {test_mse:.6f}")
        print("Best hyperparameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print("=============================================================")


if __name__ == "__main__":
    main()
