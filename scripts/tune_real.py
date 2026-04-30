#!/usr/bin/env python3
"""
MPI + Optuna hyperparameter tuning for real Goodreads data.

Objective: minimize post-update train MSE (projected rating space −2..+2).

Usage (example):
    mpirun -n 16 python scripts/tune_real.py \
        --train-dir data/real/cos090/train \
        --test-dir  data/real/cos090/test \
        --book-topics-npz data/real/book_topics.npz \
        --model KF-AF --n-trials 30000 --min-rating-diff 2.0 \
        --eval-min-cosine 0.90 \
        --output-dir results/real/tune/KF-AF
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from mpi4py import MPI
import optuna

from nspb.posterior_distances import kl_divergence, nig_kl_divergence
from nspb.models import (
    BayesianModel,
    VarianceBoundedBayesianModel,
    BayesianForgettingFactorModel,
    AROW_Regression,
    KalmanFilter,
    BayesianSlidingWindowModel,
    PowerPriorBayesianModel,
    NormalInverseGammaModel,
)


# -------------------------
# IO helpers
# -------------------------
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _list_json_files(d: str) -> List[str]:
    dd = Path(d)
    files = sorted(str(p) for p in dd.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    if not files:
        raise SystemExit(f"No JSON files found in {dd}")
    return files


# -------------------------
# Numeric helpers
# -------------------------
def _ensure_timesteps(reviews: List[Dict[str, Any]]) -> None:
    for i, r in enumerate(reviews, start=1):
        if not isinstance(r.get("timestep"), int):
            r["timestep"] = i

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _rating_scaled(r: float) -> float:
    return r - 3.0 if 0.5 <= r <= 5.5 else r

def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rxm, rym = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rxm * rxm).sum() * (rym * rym).sum()))
    return float((rxm * rym).sum() / denom) if denom else float("nan")

def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float(((labels == 1) * ranks).sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def _pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1).astype(float)
    fp = np.cumsum(y == 0).astype(float)
    prec = tp / np.maximum(tp + fp, 1)
    return float(prec[y == 1].sum() / n_pos)


# -------------------------
# Topic loading
# -------------------------
def _collect_book_ids(files: List[str]) -> Set[str]:
    needed: Set[str] = set()
    for fp in files:
        try:
            data = _load_json(fp)
        except Exception:
            continue
        for r in data.get("reviews", []):
            bid = r.get("book_id")
            if bid is not None:
                needed.add(str(bid))
    return needed

def _load_topics(npz_path: str, needed: Set[str]) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    book_ids, topics = data["book_ids"], data["topics"]
    idx = {str(b): i for i, b in enumerate(book_ids)}
    return {bid: topics[idx[bid]] for bid in needed if bid in idx}


# -------------------------
# Model construction
# -------------------------
MODEL_CHOICES = ["KF-AF", "AROW", "BLR", "BLR-VB", "BLR-FF", "BLR-SW", "BLR-PP", "BLR-NIG"]

PAPER_TO_KEY = {
    "KF-AF":   "KF",
    "BLR-VB":  "vbBLR",
    "BLR-FF":  "fBLR",
    "BLR-SW":  "BLRsw",
    "BLR-PP":  "PBLR",
    "BLR-NIG": "NIG",
}

def _build_model(key: str, d: int, p: Dict[str, float]):
    if key == "KF":
        return KalmanFilter(d, p.get("variance_p", 3.0), p.get("variance", 0.3),
                            p.get("delta", 0.015), p.get("eta", 1e-5))
    if key == "BLR":
        return BayesianModel(d, p.get("default_variance", 50.0), p.get("noise_precision", 0.33))
    if key == "vbBLR":
        return VarianceBoundedBayesianModel(d, p.get("default_variance", 15.0),
                                            p.get("noise_precision", 0.33), p.get("tau_v", 1.0))
    if key == "fBLR":
        return BayesianForgettingFactorModel(d, p.get("default_variance", 50.0),
                                             p.get("noise_precision", 0.33))
    if key == "AROW":
        return AROW_Regression(d, p.get("lam1", 0.4), p.get("lam2", 0.2))
    if key == "BLRsw":
        return BayesianSlidingWindowModel(d, int(p.get("m", 20)),
                                          p.get("default_variance", 50.0),
                                          p.get("noise_precision", 0.33))
    if key == "PBLR":
        return PowerPriorBayesianModel(d, p.get("alpha", 0.9),
                                       p.get("default_variance", 50.0),
                                       p.get("noise_precision", 0.33))
    if key == "NIG":
        return NormalInverseGammaModel(d, p.get("default_variance", 50.0),
                                       p.get("default_a", 2.0), p.get("default_b", 2.0))
    raise ValueError(f"Unknown model key: {key}")

def _suggest_params(trial: optuna.Trial, key: str) -> Dict[str, float]:
    if key == "KF":
        return {"variance_p": trial.suggest_float("variance_p", 1e-2, 5e1, log=True),
                "variance":   trial.suggest_float("variance",   1e-2, 5e1, log=True),
                "delta":      trial.suggest_float("delta",      1e-3, 0.3, log=True),
                "eta":        trial.suggest_float("eta",        1e-7, 1e-1, log=True)}
    if key == "BLR":
        return {"default_variance":  trial.suggest_float("default_variance",  1e-2, 1e2, log=True),
                "noise_precision":   trial.suggest_float("noise_precision",   1e-2, 3e1, log=True)}
    if key == "vbBLR":
        return {"default_variance":  trial.suggest_float("default_variance",  1e-2, 1e2, log=True),
                "noise_precision":   trial.suggest_float("noise_precision",   1e-2, 3e1, log=True),
                "tau_v":             trial.suggest_float("tau_v",             0.3,  1.0, log=True)}
    if key == "fBLR":
        return {"default_variance":  trial.suggest_float("default_variance",  1e-2, 5e1, log=True),
                "noise_precision":   trial.suggest_float("noise_precision",   1e-1, 3e1, log=True),
                "rho":               trial.suggest_float("rho",               0.95, 0.999)}
    if key == "AROW":
        return {"lam1": trial.suggest_float("lam1", 1e-2, 1e2, log=True),
                "lam2": trial.suggest_float("lam2", 1e-2, 1e2, log=True)}
    if key == "BLRsw":
        return {"default_variance":  trial.suggest_float("default_variance",  1e-2, 1e2, log=True),
                "noise_precision":   trial.suggest_float("noise_precision",   1e-2, 3e1, log=True),
                "m":                 trial.suggest_int("m", 2, 50)}
    if key == "PBLR":
        return {"alpha":             trial.suggest_float("alpha",             0.85, 1.0),
                "default_variance":  trial.suggest_float("default_variance",  1e-2, 1e2, log=True),
                "noise_precision":   trial.suggest_float("noise_precision",   1e-2, 3e1, log=True)}
    if key == "NIG":
        return {"default_variance":  trial.suggest_float("default_variance",  1e-2, 1e2, log=True),
                "default_a":         trial.suggest_float("default_a",         1.01, 1e2, log=True),
                "default_b":         trial.suggest_float("default_b",         1e-3, 1e2, log=True)}
    raise ValueError(f"Unknown model key for search space: {key}")


# -------------------------
# Per-user MSE (train, post-update)
# -------------------------
def _user_sse(path: str, book_to_topic: Dict[str, np.ndarray],
              key: str, params: Dict[str, float]) -> Tuple[float, int]:
    data = _load_json(path)
    reviews = data.get("reviews", [])
    if not isinstance(reviews, list) or not reviews:
        return 0.0, 0
    _ensure_timesteps(reviews)
    reviews = sorted(reviews, key=lambda r: int(r.get("timestep", 0)))

    local = {str(r.get("book_id")): book_to_topic[str(r.get("book_id"))]
             for r in reviews if str(r.get("book_id")) in book_to_topic}
    if not local:
        return 0.0, 0

    d = next(iter(local.values())).shape[0]
    model = _build_model(key, d, params)
    sse, n = 0.0, 0
    for r in reviews:
        x = local.get(str(r.get("book_id")))
        if x is None:
            continue
        rating = _safe_float(r.get("rating"))
        if rating is None:
            continue
        y = _rating_scaled(rating)
        try:
            if key == "fBLR":
                model.update(x, y, rho=float(params.get("rho", 0.98)))
            else:
                model.update(x, y)
            y_hat = float(model.predict(x))
        except Exception:
            continue
        if math.isfinite(y_hat):
            sse += (y_hat - y) ** 2
            n += 1
    return sse, n


# -------------------------
# Per-user full metrics (for post-tuning eval)
# -------------------------
def _parse_pair_timesteps(rj: Dict[str, Any]) -> List[int]:
    pts = rj.get("pair_k_timesteps", [])
    return [t for t in pts if isinstance(t, int)] if isinstance(pts, list) else []

def _extract_cosine(rj: Dict[str, Any], kt: int,
                    details: Dict[int, Dict[str, Any]], pair_ts: List[int]) -> Optional[float]:
    d = details.get(kt)
    if isinstance(d, dict):
        for k in ("cosine_sim_with_j", "cosine_sim", "pair_cosine", "cos_sim"):
            v = _safe_float(d.get(k))
            if v is not None:
                return v
    cos_list = rj.get("pair_k_cosine_sim")
    if isinstance(cos_list, list):
        try:
            idx = pair_ts.index(kt)
            return _safe_float(cos_list[idx]) if idx < len(cos_list) else None
        except ValueError:
            return None
    return None

def _user_full_metrics(path: str, book_to_topic: Dict[str, np.ndarray],
                       key: str, params: Dict[str, float],
                       min_rating_diff: float, min_cosine: float,
                       mse_mode: str) -> Dict[str, Any]:
    data = _load_json(path)
    reviews = data.get("reviews", [])
    if not isinstance(reviews, list) or not reviews:
        return {"ok": False}
    _ensure_timesteps(reviews)
    reviews = sorted(reviews, key=lambda r: int(r.get("timestep", 0)))

    local = {str(r.get("book_id")): book_to_topic[str(r.get("book_id"))]
             for r in reviews if str(r.get("book_id")) in book_to_topic}
    if not local:
        return {"ok": False}

    d = next(iter(local.values())).shape[0]
    model = _build_model(key, d, params)
    fblr_rho = float(params.get("rho", 0.98)) if key == "fBLR" else None

    state: Dict[int, Any] = {}
    sse, n_pred = 0.0, 0

    for r in reviews:
        t = int(r["timestep"])
        x = local.get(str(r.get("book_id")))
        if x is None:
            continue
        rating = _safe_float(r.get("rating"))
        if rating is None:
            continue
        y = _rating_scaled(rating)

        try:
            if mse_mode == "pre":
                y_hat = float(model.predict(x))
                if math.isfinite(y_hat):
                    sse += (y_hat - y) ** 2
                    n_pred += 1
                upd = model.update(x, y) if key != "fBLR" else model.update(x, y, rho=fblr_rho)
            else:
                upd = model.update(x, y) if key != "fBLR" else model.update(x, y, rho=fblr_rho)
                y_hat = float(model.predict(x))
                if math.isfinite(y_hat):
                    sse += (y_hat - y) ** 2
                    n_pred += 1
        except Exception:
            continue

        if key == "NIG":
            if isinstance(upd, tuple) and len(upd) == 4:
                mu, V, a, b = upd
                state[t] = {"mu": np.asarray(mu).reshape(-1), "V": np.asarray(V),
                             "a": float(a), "b": float(b)}
        else:
            if isinstance(upd, tuple) and len(upd) == 3:
                mu, _, S = upd
            elif isinstance(upd, tuple) and len(upd) == 2:
                mu, S = upd
            else:
                continue
            mu = np.asarray(mu, dtype=np.float64).reshape(-1)
            S = np.asarray(S, dtype=np.float64)
            if S.ndim != 2 or S.shape != (mu.size, mu.size):
                S = np.eye(mu.size)
            state[t] = (mu, S)

    rating_by_t = {}
    for r in reviews:
        v = _safe_float(r.get("rating"))
        if v is not None:
            rating_by_t[int(r["timestep"])] = v

    xs, ys, auc_labels, auc_scores = [], [], [], []
    kl_stable, kl_change = [], []

    for rj in reviews:
        jt = int(rj["timestep"])
        pair_ts = _parse_pair_timesteps(rj)
        if not pair_ts or jt not in state:
            continue
        details = {int(d["k_timestep"]): d for d in rj.get("pair_k_details", [])
                   if isinstance(d, dict) and isinstance(d.get("k_timestep"), int)}
        rj_rating = _safe_float(rj.get("rating"))
        if rj_rating is None:
            continue
        for kt in pair_ts:
            if kt not in state or kt not in rating_by_t:
                continue
            cs = _extract_cosine(rj, kt, details, pair_ts)
            if cs is None or not math.isfinite(cs) or cs <= min_cosine:
                continue
            dr = abs(rj_rating - rating_by_t[kt])
            if not math.isfinite(dr) or dr < min_rating_diff:
                continue

            try:
                if key == "NIG":
                    sj, sk = state[jt], state[kt]
                    kl_val = float(nig_kl_divergence(V_t=sj["V"], V_tp1=sk["V"],
                                                     mu_t=sj["mu"], mu_tp1=sk["mu"],
                                                     a_t=sj["a"], a_tp1=sk["a"],
                                                     b_t=sj["b"], b_tp1=sk["b"]))
                else:
                    mu_j, S_j = state[jt]
                    mu_k, S_k = state[kt]
                    kl_val = float(np.asarray(kl_divergence(S_k, S_j,
                                                             mu_k.reshape(-1, 1),
                                                             mu_j.reshape(-1, 1))).reshape(-1)[0])
            except Exception:
                continue

            if not math.isfinite(kl_val) or kl_val < 0:
                continue

            xs.append(float(dr))
            ys.append(kl_val)
            dr_int = int(round(dr))
            if dr_int == 0:
                auc_labels.append(0); auc_scores.append(kl_val); kl_stable.append(kl_val)
            elif dr_int > 1:
                auc_labels.append(1); auc_scores.append(kl_val); kl_change.append(kl_val)

    return {
        "ok": True,
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "auc_labels": np.asarray(auc_labels, dtype=np.int32),
        "auc_scores": np.asarray(auc_scores, dtype=np.float64),
        "kl_stable": np.asarray(kl_stable, dtype=np.float64),
        "kl_change": np.asarray(kl_change, dtype=np.float64),
        "sse": float(sse),
        "n_pred": int(n_pred),
    }


# -------------------------
# MPI aggregation
# -------------------------
def _run_mpi(comm, files: List[str], key: str, params: Dict[str, float],
             book_to_topic: Dict[str, np.ndarray],
             min_rating_diff: float, min_cosine: float,
             mse_mode: str) -> Dict[str, Any]:
    rank = comm.Get_rank()
    x_loc, y_loc, al_loc, as_loc, ks_loc, kc_loc = [], [], [], [], [], []
    sse_loc, n_loc, n_users = 0.0, 0, 0

    for f in files:
        res = _user_full_metrics(f, book_to_topic, key, params,
                                 min_rating_diff, min_cosine, mse_mode)
        if not res.get("ok"):
            continue
        n_users += 1
        sse_loc += res["sse"]; n_loc += res["n_pred"]
        for arr, lst in [(res["x"], x_loc), (res["y"], y_loc),
                         (res["auc_labels"], al_loc), (res["auc_scores"], as_loc),
                         (res["kl_stable"], ks_loc), (res["kl_change"], kc_loc)]:
            if arr.size:
                lst.append(arr)

    gathered = comm.gather((x_loc, y_loc, al_loc, as_loc, ks_loc, kc_loc,
                            sse_loc, n_loc, n_users), root=0)
    if rank != 0:
        return {}

    def cat(idx, dtype=np.float64):
        arrs = [a for g in gathered for a in g[idx] if isinstance(a, np.ndarray) and a.size]
        return np.concatenate(arrs).astype(dtype) if arrs else np.empty(0, dtype)

    x_g, y_g = cat(0), cat(1)
    al_g = cat(2, np.int32).astype(np.int64)
    as_g = cat(3)
    ks_g, kc_g = cat(4), cat(5)
    sse_t = sum(g[6] for g in gathered)
    n_t = sum(g[7] for g in gathered)
    nu = sum(g[8] for g in gathered)

    rho = _spearman(x_g, y_g) if x_g.size else float("nan")
    roc = _roc_auc(al_g, as_g) if al_g.size else float("nan")
    pr = _pr_auc(al_g, as_g) if al_g.size else float("nan")

    def _med(a): return float(np.median(a)) if a.size else float("nan")
    def _mn(a): return float(np.mean(a)) if a.size else float("nan")

    mks, mkc = _med(ks_g), _med(kc_g)
    ratio_med = float(mkc / mks) if math.isfinite(mkc) and math.isfinite(mks) and mks > 0 else float("nan")
    mks2, mkc2 = _mn(ks_g), _mn(kc_g)
    ratio_mean = float(mkc2 / mks2) if math.isfinite(mkc2) and math.isfinite(mks2) and mks2 > 0 else float("nan")

    return {
        "rho": rho, "roc_auc": roc, "pr_auc": pr,
        "median_kl_stable": mks, "median_kl_change": mkc,
        "mean_kl_stable": mks2, "mean_kl_change": mkc2,
        "median_ratio": ratio_med, "mean_ratio": ratio_mean,
        "mse": float(sse_t / n_t) if n_t > 0 else float("nan"),
        "n_pred": int(n_t), "n_users": int(nu),
        "n_pairs_corr": int(x_g.size), "n_pairs_auc": int(al_g.size),
        "n_auc_pos": int((al_g == 1).sum()), "n_auc_neg": int((al_g == 0).sum()),
        "mse_mode": mse_mode,
    }


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--book-topics-npz", required=True)
    ap.add_argument("--model", choices=MODEL_CHOICES, required=True)
    ap.add_argument("--min-rating-diff", type=float, default=2.0)
    ap.add_argument("--eval-min-cosine", type=float, default=0.90)
    ap.add_argument("--n-trials", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-ratings-fraction", type=float, default=0.99)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    model_key = PAPER_TO_KEY.get(args.model, args.model)

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[rank0] world_size={size}, model={args.model} (key={model_key})", flush=True)
    comm.Barrier()

    # --- File lists ---
    if rank == 0:
        train_files = _list_json_files(args.train_dir)
        test_files  = _list_json_files(args.test_dir)
        print(f"[rank0] train={len(train_files)}, test={len(test_files)}", flush=True)
    else:
        train_files = test_files = None
    train_files = comm.bcast(train_files, root=0)
    test_files  = comm.bcast(test_files,  root=0)

    my_train = [train_files[i] for i in range(len(train_files)) if i % size == rank]
    my_test  = [test_files[i]  for i in range(len(test_files))  if i % size == rank]

    # --- Book topics ---
    needed_local = _collect_book_ids(my_train) | _collect_book_ids(my_test)
    needed_all = comm.gather(needed_local, root=0)
    if rank == 0:
        needed = sorted(set().union(*needed_all))
        print(f"[rank0] needed_book_ids={len(needed)}", flush=True)
    else:
        needed = None
    needed = comm.bcast(needed, root=0)
    book_to_topic = _load_topics(args.book_topics_npz, set(needed))
    comm.Barrier()

    # --- Usable count for coverage constraint ---
    usable_local = sum(
        1 for f in my_train for r in _load_json(f).get("reviews", [])
        if str(r.get("book_id", "")) in book_to_topic and _safe_float(r.get("rating")) is not None
    )
    expected_total = int(comm.allreduce(usable_local, op=MPI.SUM))
    if rank == 0:
        print(f"[rank0] usable_train_ratings={expected_total}", flush=True)

    # --- Optuna objective ---
    def objective(trial: Optional[optuna.Trial]) -> float:
        if rank == 0:
            model_params = _suggest_params(trial, model_key)
        else:
            model_params = None
        model_params = comm.bcast(model_params, root=0)

        sse_loc, n_loc = 0.0, 0
        for f in my_train:
            s, n = _user_sse(f, book_to_topic, model_key, model_params)
            sse_loc += s; n_loc += n
        gathered = comm.gather((sse_loc, n_loc), root=0)
        if rank != 0:
            return 0.0
        sse_t = sum(x for x, _ in gathered)
        n_t   = sum(n for _, n in gathered)
        min_n = int(math.ceil(args.min_ratings_fraction * expected_total))
        if n_t < min_n or n_t == 0:
            return 1e18
        trial.set_user_attr("n_ratings", int(n_t))
        return float(sse_t / n_t)

    if rank == 0:
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=args.seed))
    else:
        study = None

    for _ in range(args.n_trials):
        if rank == 0:
            trial = study.ask()
            value = objective(trial)
            study.tell(trial, value)
        else:
            objective(None)

    if rank == 0:
        best_params = dict(study.best_params)
        best_mse    = float(study.best_value)
        print(f"[rank0] best_train_mse={best_mse}", flush=True)
        print(f"[rank0] best_params={best_params}", flush=True)
    else:
        best_params = best_mse = None
    best_params = comm.bcast(best_params, root=0)
    best_mse    = comm.bcast(best_mse,    root=0)

    # --- Full evaluation with best params ---
    train_stats = _run_mpi(comm, my_train, model_key, best_params, book_to_topic,
                           args.min_rating_diff, args.eval_min_cosine, mse_mode="post")
    test_stats  = _run_mpi(comm, my_test,  model_key, best_params, book_to_topic,
                           args.min_rating_diff, args.eval_min_cosine, mse_mode="pre")

    if rank != 0:
        return

    def _print(name, st):
        print(f"\n=== {name} ===")
        for k in ("mse", "rho", "roc_auc", "pr_auc",
                  "median_kl_stable", "median_kl_change", "median_ratio",
                  "n_users", "n_pairs_corr", "n_pairs_auc"):
            print(f"  {k}: {st.get(k, float('nan'))}")

    _print("TRAIN", train_stats)
    _print("TEST",  test_stats)

    out = {
        "model": args.model,
        "model_key": model_key,
        "best_train_mse": best_mse,
        "best_hyperparameters": best_params,
        "eval_min_rating_diff": args.min_rating_diff,
        "eval_min_cosine": args.eval_min_cosine,
        "train": train_stats,
        "test": test_stats,
    }
    out_path = out_dir / "tune_results.json"
    _save_json(str(out_path), out)
    print(f"\n[rank0] Saved: {out_path}")


if __name__ == "__main__":
    main()
