#!/usr/bin/env python3
"""
Evaluate models on the real Goodreads dataset across cosine thresholds and
timestamp-gap windows, then emit per-user metrics CSVs and a summary JSON.

Per-user CSVs (one per model × cosine × gap × split) have columns:
    user_file, rho, roc_auc, pr_auc, n_pairs_corr, n_auc_pos, n_auc_neg

MSE modes:
  TRAIN MSE (post-update, same timestep):
    For obs i > warmup: update on (x_i, y_i) → predict y_hat_i = mu_i^T x_i
  TEST MSE (one-step-ahead, prequential):
    For obs i > warmup: y_hat_i = mu_{i-1}^T x_i, then update on obs i

Pair filtering:
  - min_cosine: cosine similarity between paired items
  - max_pair_gap_days: max |timestamp_k - timestamp_j| in days (or 'all')
  - min_rating_diff: minimum |rating_j - rating_k| to include a pair

Outputs (in --output-dir):
  GLOBAL__all_models_summary.json
  per_user_csv/per_user__<model>__cos<cos>__gap<gap>__<split>.csv
"""

import os
import csv
import json
import math
import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from array import array

import numpy as np

from mpi4py import MPI
from mpi4py.util import pkl5  # noqa: F401

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

MODEL_CHOICES = ["KF-AF", "AROW", "BLR", "BLR-VB", "BLR-FF", "BLR-SW", "BLR-PP", "BLR-NIG"]

PAPER_TO_KEY = {
    "KF-AF":   "KF",
    "BLR-VB":  "vbBLR",
    "BLR-FF":  "fBLR",
    "BLR-SW":  "BLRsw",
    "BLR-PP":  "PBLR",
    "BLR-NIG": "NIG",
}


# -----------------------------
# IO helpers
# -----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_json_files(d: str) -> List[str]:
    dd = Path(d)
    files = sorted([str(p) for p in dd.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    if not files:
        raise SystemExit(f"No JSON files found in {dd}")
    return files


# -----------------------------
# small utils
# -----------------------------
def ensure_timesteps(reviews: List[Dict[str, Any]]) -> None:
    for i, r in enumerate(reviews, start=1):
        if not isinstance(r.get("timestep"), int):
            r["timestep"] = i


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        try:
            vf = safe_float(x)
            if vf is None:
                return None
            return int(vf)
        except Exception:
            return None


def rating_to_scaled(r: float) -> float:
    if 0.5 <= r <= 5.5:
        return r - 3.0
    return r


def parse_pair_timesteps(rj: Dict[str, Any]) -> List[int]:
    pts = rj.get("pair_k_timesteps")
    if isinstance(pts, list):
        out: List[int] = []
        for t in pts:
            if isinstance(t, int):
                out.append(int(t))
            else:
                tf = safe_float(t)
                if tf is not None:
                    out.append(int(tf))
        return out
    single = rj.get("pair_k_timestep")
    if single is not None:
        if isinstance(single, int):
            return [int(single)]
        sf = safe_float(single)
        if sf is not None:
            return [int(sf)]
    return []


# -----------------------------
# Spearman / AUC / AP helpers
# -----------------------------
def _rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)

    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    rxm = rx - rx.mean()
    rym = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rxm * rxm) * np.sum(rym * rym)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rxm * rym) / denom)


def roc_auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0:
        return float("nan")
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average_ties(scores)
    sum_ranks_pos = float(np.sum(ranks[labels == 1]))
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(labels == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = fp = 0
    ap = 0.0
    for i in range(y.size):
        if y[i] == 1:
            tp += 1
            ap += tp / (tp + fp)
        else:
            fp += 1
    return float(ap / n_pos)


# -----------------------------
# per-user metrics
# -----------------------------
def per_user_metrics(
    x_np: np.ndarray,
    y_np: np.ndarray,
    y_auc_np: np.ndarray,
    s_auc_np: np.ndarray,
) -> Dict[str, Any]:
    rho = spearman_rho(x_np, y_np) if x_np.size >= 2 else float("nan")

    if y_auc_np.size > 0:
        y64 = np.asarray(y_auc_np, dtype=np.int64)
        n_pos = int(np.sum(y64 == 1))
        n_neg = int(np.sum(y64 == 0))
        if n_pos > 0 and n_neg > 0:
            roc = roc_auc_from_scores(y64, s_auc_np)
            pr = average_precision_from_scores(y64, s_auc_np)
        else:
            roc = pr = float("nan")
    else:
        n_pos = n_neg = 0
        roc = pr = float("nan")

    return {
        "rho": float(rho) if math.isfinite(rho) else float("nan"),
        "roc_auc": float(roc) if math.isfinite(roc) else float("nan"),
        "pr_auc": float(pr) if math.isfinite(pr) else float("nan"),
        "n_pairs_corr": int(x_np.size),
        "n_auc_pos": int(n_pos),
        "n_auc_neg": int(n_neg),
    }


# -----------------------------
# topic loading
# -----------------------------
def collect_needed_book_ids(user_json_paths: List[str]) -> Set[str]:
    needed: Set[str] = set()
    for fp in user_json_paths:
        try:
            data = load_json(fp)
        except Exception:
            continue
        for r in data.get("reviews", []):
            bid = r.get("book_id")
            if bid is not None:
                needed.add(str(bid))
    return needed


def load_book_topics_from_npz(npz_path: str, needed_book_ids: Set[str]) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    book_ids = data["book_ids"]
    topics = data["topics"]
    id_to_idx = {str(bid): i for i, bid in enumerate(book_ids)}
    out: Dict[str, np.ndarray] = {}
    for bid in needed_book_ids:
        i = id_to_idx.get(str(bid))
        if i is not None:
            out[str(bid)] = topics[i]
    return out


# -----------------------------
# model builder
# -----------------------------
def _build_model(model_key: str, n_topics: int, params: Dict[str, float]):
    if model_key == "KF":
        return KalmanFilter(
            n_topics,
            variance_p=params.get("variance_p", 0.01),
            variance=params.get("variance", 0.05),
            delta=params.get("delta", 0.0),
            eta=params.get("eta", 0.001),
        )
    if model_key == "BLR":
        return BayesianModel(
            n_topics,
            default_variance=params.get("default_variance", 50.0),
            default_noise_precision=params.get("noise_precision", 0.33),
        )
    if model_key == "vbBLR":
        return VarianceBoundedBayesianModel(
            n_topics,
            default_variance=params.get("default_variance", 50.0),
            default_noise_precision=params.get("noise_precision", 0.33),
            tau=params.get("tau_v", 1.0),
        )
    if model_key == "fBLR":
        return BayesianForgettingFactorModel(
            n_topics,
            default_variance=params.get("default_variance", 50.0),
            default_noise_precision=params.get("noise_precision", 0.33),
        )
    if model_key == "AROW":
        return AROW_Regression(
            n_topics,
            lam1=params.get("lam1", 0.4),
            lam2=params.get("lam2", 0.2),
        )
    if model_key == "BLRsw":
        return BayesianSlidingWindowModel(
            n_topics,
            m=int(params.get("window_size", 20)),
            default_variance=params.get("default_variance", 50.0),
            default_noise_precision=params.get("noise_precision", 0.33),
        )
    if model_key == "PBLR":
        return PowerPriorBayesianModel(
            n_topics,
            alpha=params.get("alpha", 0.9),
            default_variance=params.get("default_variance", 50.0),
            default_noise_precision=params.get("noise_precision", 0.33),
        )
    if model_key == "NIG":
        return NormalInverseGammaModel(
            n_topics,
            default_variance=params.get("default_variance", 50.0),
            default_a=params.get("default_a", 2.0),
            default_b=params.get("default_b", 2.0),
        )
    raise ValueError(f"Unknown model_key={model_key!r}")


# -----------------------------
# cosine extraction
# -----------------------------
def extract_pair_cosine_fast(
    rj: Dict[str, Any],
    kt: int,
    details_by_k: Dict[int, Dict[str, Any]],
    idx_by_kt: Dict[int, int],
) -> Optional[float]:
    d = details_by_k.get(int(kt))
    if isinstance(d, dict):
        for key in ("cosine_sim_with_j", "cosine", "cosine_sim", "pair_cosine", "cos_sim", "cosine_similarity"):
            if key in d:
                v = safe_float(d.get(key))
                if v is not None:
                    return v

    cos_list = rj.get("pair_cosines")
    if not isinstance(cos_list, list):
        cos_list = rj.get("pair_k_cosine_sim")
    if isinstance(cos_list, list):
        idx = idx_by_kt.get(int(kt))
        if idx is not None and 0 <= idx < len(cos_list):
            return safe_float(cos_list[idx])

    if rj.get("pair_cosine") is not None:
        return safe_float(rj.get("pair_cosine"))

    return None


# -----------------------------
# per-user processing
# -----------------------------
def process_one_user(
    paired_path: str,
    book_to_topic: Dict[str, np.ndarray],
    model_key: str,
    model_params: Dict[str, float],
    min_rating_diff: float,
    min_cosine: float,
    max_pair_time_gap_seconds: Optional[int],
    mse_mode: str,
    warmup: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int], float, int, int]:
    if mse_mode not in ("train_same_timestep", "test_one_step_ahead"):
        raise ValueError(f"Unknown mse_mode={mse_mode!r}")

    paired = load_json(paired_path)
    reviews_raw = paired.get("reviews", [])
    _empty = (
        np.empty(0, np.float64), np.empty(0, np.float64),
        np.empty(0, np.int32), np.empty(0, np.float64),
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}, 0.0, 0, 0,
    )
    if not isinstance(reviews_raw, list) or not reviews_raw:
        return _empty

    ensure_timesteps(reviews_raw)

    local_book_to_topic: Dict[str, np.ndarray] = {}
    for r in reviews_raw:
        bid = r.get("book_id")
        if bid is None:
            continue
        bid = str(bid)
        vec = book_to_topic.get(bid)
        if vec is not None:
            local_book_to_topic[bid] = vec
    if not local_book_to_topic:
        return _empty

    n_topics = int(next(iter(local_book_to_topic.values())).shape[0])
    model = _build_model(model_key, n_topics, model_params)

    sse = 0.0
    n_pred = 0
    prev_mu: Optional[np.ndarray] = None
    obs_idx = 0
    state_by_timestep: Dict[int, Dict[str, Any]] = {}

    reviews_raw = sorted(reviews_raw, key=lambda r: int(r.get("timestep", 0)))

    for r in reviews_raw:
        t = int(r["timestep"])
        bid = str(r.get("book_id"))
        topic = local_book_to_topic.get(bid)
        if topic is None:
            continue

        r_raw = safe_float(r.get("rating"))
        if r_raw is None:
            continue

        y_scaled = float(rating_to_scaled(r_raw))
        obs_idx += 1

        if mse_mode == "test_one_step_ahead" and obs_idx > warmup and prev_mu is not None:
            try:
                y_pred = float(np.dot(prev_mu, np.asarray(topic).reshape(-1)))
                if math.isfinite(y_pred) and math.isfinite(y_scaled):
                    err = y_pred - y_scaled
                    sse += err * err
                    n_pred += 1
            except Exception:
                pass

        if model_key == "vbBLR":
            out = model.update(topic, y_scaled)
            if isinstance(out, tuple) and len(out) == 3:
                mu, _, S = out
            else:
                mu, S = out
            mu_col = np.asarray(mu).reshape(-1, 1)
            state_by_timestep[t] = {"mu": mu_col, "S": np.asarray(S)}
            mu_vec = mu_col.reshape(-1).astype(np.float64, copy=False)

        elif model_key == "fBLR":
            rho_val = float(model_params.get("rho", 0.98))
            mu, S = model.update(topic, y_scaled, rho=rho_val)
            mu_vec = np.asarray(mu, dtype=np.float64).reshape(-1)
            state_by_timestep[t] = {"mu": mu_vec.reshape(-1, 1), "S": np.asarray(S)}

        elif model_key == "NIG":
            mu, V, a, b = model.update(topic, y_scaled)
            mu_vec = np.asarray(mu, dtype=np.float64).reshape(-1)
            V_arr = np.asarray(V, dtype=np.float64)
            if V_arr.ndim != 2 or V_arr.shape[0] != V_arr.shape[1] or V_arr.shape[0] != mu_vec.size:
                V_arr = np.eye(mu_vec.size, dtype=np.float64)
            invV_arr = np.asarray(getattr(model, "invV", np.linalg.pinv(V_arr)), dtype=np.float64)
            if invV_arr.shape != V_arr.shape:
                invV_arr = np.linalg.pinv(V_arr)
            state_by_timestep[t] = {
                "mu": mu_vec, "V": V_arr,
                "a": float(a), "b": float(b), "invV": invV_arr,
            }
            mu_vec = mu_vec

        else:
            mu, S = model.update(topic, y_scaled)
            mu_col = np.asarray(mu).reshape(-1, 1)
            state_by_timestep[t] = {"mu": mu_col, "S": np.asarray(S)}
            mu_vec = mu_col.reshape(-1).astype(np.float64, copy=False)

        if mse_mode == "train_same_timestep" and obs_idx > warmup:
            try:
                y_pred = float(np.dot(mu_vec, np.asarray(topic).reshape(-1)))
                if math.isfinite(y_pred) and math.isfinite(y_scaled):
                    err = y_pred - y_scaled
                    sse += err * err
                    n_pred += 1
            except Exception:
                pass

        prev_mu = mu_vec

    rating_by_t: Dict[int, float] = {}
    timestamp_by_t: Dict[int, int] = {}
    for r in reviews_raw:
        tt = int(r["timestep"])
        rr = safe_float(r.get("rating"))
        if rr is not None:
            rating_by_t[tt] = float(rr)
        ts = safe_int(r.get("timestamp"))
        if ts is not None:
            timestamp_by_t[tt] = int(ts)

    x_arr: array = array("d")
    y_arr: array = array("d")
    y_auc_arr: array = array("i")
    s_auc_arr: array = array("d")

    dr_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    used_pairs = 0

    max_gap: Optional[int] = None
    if max_pair_time_gap_seconds is not None:
        mg = int(max_pair_time_gap_seconds)
        if mg > 0:
            max_gap = mg

    for rj in reviews_raw:
        jt = int(rj["timestep"])
        pair_ts = parse_pair_timesteps(rj)
        if not pair_ts:
            continue

        idx_by_kt = {int(tt): i for i, tt in enumerate(pair_ts)}

        details_by_k: Dict[int, Dict[str, Any]] = {}
        if isinstance(rj.get("pair_k_details"), list):
            for d in rj["pair_k_details"]:
                if isinstance(d, dict) and d.get("k_timestep") is not None:
                    try:
                        details_by_k[int(d["k_timestep"])] = d
                    except Exception:
                        pass

        rj_rating = rating_by_t.get(jt)
        if rj_rating is None:
            continue
        sj = state_by_timestep.get(jt)
        if sj is None:
            continue
        ts_j = timestamp_by_t.get(jt)

        for kt in pair_ts:
            kt = int(kt)
            rk_rating = rating_by_t.get(kt)
            if rk_rating is None:
                continue
            sk = state_by_timestep.get(kt)
            if sk is None:
                continue

            if max_gap is not None:
                ts_k = timestamp_by_t.get(kt)
                if ts_j is None or ts_k is None:
                    continue
                if abs(int(ts_k) - int(ts_j)) > max_gap:
                    continue

            cs = extract_pair_cosine_fast(rj, kt, details_by_k, idx_by_kt)
            if cs is None or not math.isfinite(cs) or cs <= min_cosine:
                continue

            dr = abs(float(rj_rating) - float(rk_rating))
            if not math.isfinite(dr) or dr < min_rating_diff:
                continue

            dr_int = max(0, min(4, int(round(dr))))
            dr_counts[dr_int] = dr_counts.get(dr_int, 0) + 1

            if model_key == "NIG":
                try:
                    kl_val = float(nig_kl_divergence(
                        V_t=sj["V"], V_tp1=sk["V"],
                        mu_t=sj["mu"], mu_tp1=sk["mu"],
                        a_t=sj["a"], a_tp1=sk["a"],
                        b_t=sj["b"], b_tp1=sk["b"],
                        invV_t=sj.get("invV"),
                    ))
                except Exception:
                    continue
                if not math.isfinite(kl_val):
                    try:
                        kl_val = float(nig_kl_divergence(
                            V_t=sj["V"], V_tp1=sk["V"],
                            mu_t=sj["mu"], mu_tp1=sk["mu"],
                            a_t=sj["a"], a_tp1=sk["a"],
                            b_t=sj["b"], b_tp1=sk["b"],
                            invV_t=None,
                        ))
                    except Exception:
                        continue
            else:
                mu_j = sj["mu"].reshape(-1, 1)
                S_j = sj["S"]
                mu_k = sk["mu"].reshape(-1, 1)
                S_k = sk["S"]
                kl_val = float(kl_divergence(S_k, S_j, mu_k, mu_j)[0][0])

            if not math.isfinite(kl_val) or kl_val < 0:
                continue

            x_arr.append(float(dr))
            y_arr.append(float(kl_val))

            if dr_int == 0:
                y_auc_arr.append(0)
                s_auc_arr.append(float(kl_val))
            elif dr_int > 1:
                y_auc_arr.append(1)
                s_auc_arr.append(float(kl_val))
            # dr_int == 1 excluded from AUC

            used_pairs += 1

    x_np = np.frombuffer(x_arr, dtype=np.float64)
    y_np = np.frombuffer(y_arr, dtype=np.float64)
    y_auc_np = np.frombuffer(y_auc_arr, dtype=np.int32)
    s_auc_np = np.frombuffer(s_auc_arr, dtype=np.float64)

    return x_np, y_np, y_auc_np, s_auc_np, dr_counts, float(sse), int(n_pred), int(used_pairs)


# -----------------------------
# MPI evaluation for one split
# -----------------------------
def run_full_eval_split_mpi(
    comm: MPI.Comm,
    my_files: List[str],
    model_key: str,
    model_params: Dict[str, float],
    min_rating_diff: float,
    min_cosine: float,
    max_pair_time_gap_seconds: Optional[int],
    book_to_topic: Dict[str, np.ndarray],
    mse_mode: str,
    warmup: int = 10,
) -> Dict[str, Any]:
    rank = comm.Get_rank()

    xs_local: List[np.ndarray] = []
    ys_local: List[np.ndarray] = []
    y_auc_local: List[np.ndarray] = []
    s_auc_local: List[np.ndarray] = []
    dr_counts_local: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    sse_local = 0.0
    n_pred_local = 0
    n_users_used_local = 0
    per_user_rows_local: List[Dict[str, Any]] = []

    for fp in my_files:
        try:
            x_np, y_np, y_auc_np, s_auc_np, dr_counts, sse_u, n_pred_u, used_pairs = process_one_user(
                paired_path=fp,
                book_to_topic=book_to_topic,
                model_key=model_key,
                model_params=model_params,
                min_rating_diff=min_rating_diff,
                min_cosine=min_cosine,
                max_pair_time_gap_seconds=max_pair_time_gap_seconds,
                mse_mode=mse_mode,
                warmup=warmup,
            )

            if used_pairs > 0 or n_pred_u > 0:
                n_users_used_local += 1

            if x_np.size:
                xs_local.append(x_np)
                ys_local.append(y_np)
            if y_auc_np.size:
                y_auc_local.append(y_auc_np)
                s_auc_local.append(s_auc_np)

            for k in dr_counts_local:
                dr_counts_local[k] += int(dr_counts.get(k, 0))

            sse_local += float(sse_u)
            n_pred_local += int(n_pred_u)

            um = per_user_metrics(x_np, y_np, y_auc_np, s_auc_np)
            um["user_file"] = os.path.basename(fp)
            per_user_rows_local.append(um)

        except Exception:
            continue

    xs_all = comm.gather(xs_local, root=0)
    ys_all = comm.gather(ys_local, root=0)
    y_auc_all = comm.gather(y_auc_local, root=0)
    s_auc_all = comm.gather(s_auc_local, root=0)
    dr_counts_all = comm.gather(dr_counts_local, root=0)
    sse_all = comm.gather(float(sse_local), root=0)
    n_pred_all = comm.gather(int(n_pred_local), root=0)
    n_users_all = comm.gather(int(n_users_used_local), root=0)
    per_user_rows_all = comm.gather(per_user_rows_local, root=0)

    if rank != 0:
        return {}

    def _cat(list_of_lists, dtype) -> np.ndarray:
        arrs = [a.astype(dtype, copy=False) for lst in list_of_lists for a in lst if isinstance(a, np.ndarray) and a.size]
        return np.concatenate(arrs) if arrs else np.empty(0, dtype=dtype)

    x_global = _cat(xs_all, np.float64)
    y_global = _cat(ys_all, np.float64)
    y_auc_global = _cat(y_auc_all, np.int32).astype(np.int64, copy=False)
    s_auc_global = _cat(s_auc_all, np.float64)

    rho = spearman_rho(x_global, y_global)
    roc_auc = roc_auc_from_scores(y_auc_global, s_auc_global) if y_auc_global.size else float("nan")
    pr_auc = average_precision_from_scores(y_auc_global, s_auc_global) if y_auc_global.size else float("nan")

    stable_kls = s_auc_global[y_auc_global == 0] if y_auc_global.size else np.empty(0, np.float64)
    change_kls = s_auc_global[y_auc_global == 1] if y_auc_global.size else np.empty(0, np.float64)

    median_kl_stable = float(np.median(stable_kls)) if stable_kls.size else float("nan")
    median_kl_change = float(np.median(change_kls)) if change_kls.size else float("nan")
    mean_kl_stable = float(np.mean(stable_kls)) if stable_kls.size else float("nan")
    mean_kl_change = float(np.mean(change_kls)) if change_kls.size else float("nan")

    def _ratio(num, den):
        return float(num / den) if math.isfinite(num) and math.isfinite(den) and den > 0 else float("nan")

    dr_counts_total: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for dct in dr_counts_all:
        for k in dr_counts_total:
            dr_counts_total[k] += int(dct.get(k, 0))

    sse_total = float(np.sum(np.asarray(sse_all, dtype=np.float64)))
    n_pred_total = int(np.sum(np.asarray(n_pred_all, dtype=np.int64)))
    mse = float(sse_total / n_pred_total) if n_pred_total > 0 else float("nan")
    n_users_used = int(np.sum(np.asarray(n_users_all, dtype=np.int64)))

    n_auc_pos = int(np.sum(y_auc_global == 1)) if y_auc_global.size else 0
    n_auc_neg = int(np.sum(y_auc_global == 0)) if y_auc_global.size else 0

    per_user_rows_flat: List[Dict[str, Any]] = [row for lst in per_user_rows_all for row in lst]

    return {
        "mse": mse,
        "n_pred": n_pred_total,
        "mse_mode": mse_mode,
        "mse_warmup": int(warmup),
        "min_cosine": float(min_cosine),
        "max_pair_time_gap_seconds": int(max_pair_time_gap_seconds) if max_pair_time_gap_seconds is not None else None,
        "rho": float(rho) if math.isfinite(rho) else float("nan"),
        "roc_auc": float(roc_auc) if math.isfinite(roc_auc) else float("nan"),
        "pr_auc": float(pr_auc) if math.isfinite(pr_auc) else float("nan"),
        "median_kl_stable": median_kl_stable,
        "median_kl_change": median_kl_change,
        "mean_kl_stable": mean_kl_stable,
        "mean_kl_change": mean_kl_change,
        "median_ratio_change_over_stable": _ratio(median_kl_change, median_kl_stable),
        "mean_ratio_change_over_stable": _ratio(mean_kl_change, mean_kl_stable),
        "n_pairs_corr": int(x_global.size),
        "n_pairs_auc": int(y_auc_global.size),
        "n_auc_pos": n_auc_pos,
        "n_auc_neg": n_auc_neg,
        "n_users_used": n_users_used,
        "delta_r_pair_counts": dr_counts_total,
        "per_user_rows": per_user_rows_flat,
    }


# -----------------------------
# output helpers
# -----------------------------
def print_block(title: str, st: Dict[str, Any]) -> None:
    print(f"\n==================== {title} (rank0) ====================")
    print(f"MSE:                  {st.get('mse', float('nan'))}   (n_pred={st.get('n_pred', 0)})")
    print(f"MSE mode:             {st.get('mse_mode')} (warmup={st.get('mse_warmup', 'n/a')})")
    print(f"min_cosine:           {st.get('min_cosine', float('nan'))}")
    print(f"max_pair_time_gap_s:  {st.get('max_pair_time_gap_seconds', None)}")
    print(f"Spearman (rho):       {st.get('rho', float('nan'))}")
    print(f"ROC-AUC (Δ=0 vs Δ>1):{st.get('roc_auc', float('nan'))}")
    print(f"PR-AUC  (Δ=0 vs Δ>1):{st.get('pr_auc', float('nan'))}")
    print(f"Median KL stable:     {st.get('median_kl_stable', float('nan'))}")
    print(f"Median KL change:     {st.get('median_kl_change', float('nan'))}")
    print(f"Median ratio ch/st:   {st.get('median_ratio_change_over_stable', float('nan'))}")
    print(f"Mean KL stable:       {st.get('mean_kl_stable', float('nan'))}")
    print(f"Mean KL change:       {st.get('mean_kl_change', float('nan'))}")
    print(f"Mean ratio ch/st:     {st.get('mean_ratio_change_over_stable', float('nan'))}")
    print(f"n_pairs_corr:         {st.get('n_pairs_corr', 0)}")
    print(f"n_pairs_auc:          {st.get('n_pairs_auc', 0)} (pos={st.get('n_auc_pos', 0)}, neg={st.get('n_auc_neg', 0)})")
    print(f"n_users_used:         {st.get('n_users_used', 0)}")
    print(f"delta_r_pair_counts:  {st.get('delta_r_pair_counts', {})}")


def write_per_user_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["user_file", "rho", "roc_auc", "pr_auc", "n_pairs_corr", "n_auc_pos", "n_auc_neg"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


# -----------------------------
# main
# -----------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(
        description="Evaluate online learners on real Goodreads data (MPI-parallel)."
    )
    p.add_argument("--train-dir", required=True, help="Directory of per-user train JSON files")
    p.add_argument("--test-dir", required=True, help="Directory of per-user test JSON files")
    p.add_argument("--output-dir", required=True, help="Where to write GLOBAL summary JSON and per-user CSVs")
    p.add_argument("--book-topics-npz", required=True, help="NPZ file with book_ids and topics arrays")
    p.add_argument("--params-by-model-json", required=True,
                   help="Path to JSON file (or inline JSON) mapping model→hyperparams dict")
    p.add_argument("--min-rating-diff", type=float, default=2.0,
                   help="Minimum |rating_j - rating_k| to include a pair (default: 2.0)")
    p.add_argument("--warmup", type=int, default=10,
                   help="Number of warmup observations per user before computing MSE (default: 10)")
    p.add_argument("--min-cosines", default="0.90,0.95",
                   help="Comma-separated cosine thresholds (default: '0.90,0.95')")
    p.add_argument("--max-pair-gap-days", default="all,7,14",
                   help="Comma-separated max time gap in days between paired items; 'all' means no filter (default: 'all,7,14')")
    p.add_argument("--splits", default="train,test",
                   help="Comma-separated splits to run: 'train', 'test', or 'train,test' (default: 'train,test')")
    p.add_argument("--models", default="",
                   help=f"Comma-separated model list (default: all). Choices: {','.join(MODEL_CHOICES)}")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    out_dir = Path(args.output_dir)
    per_user_dir = out_dir / "per_user_csv"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        per_user_dir.mkdir(parents=True, exist_ok=True)
        print(f"[rank0] world_size={size}", flush=True)
        print(f"[rank0] warmup={args.warmup}", flush=True)
    comm.Barrier()

    # parse splits
    split_tokens = [tok.strip().lower() for tok in args.splits.split(",") if tok.strip()]
    if not split_tokens or any(tok not in {"train", "test"} for tok in split_tokens):
        raise SystemExit("--splits must be 'train', 'test', or 'train,test'")
    splits: List[str] = []
    for tok in split_tokens:
        if tok not in splits:
            splits.append(tok)

    # parse models — accept paper names (KF-AF, BLR-VB, …) and normalize to internal keys
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models.strip() else list(MODEL_CHOICES)
    models = [PAPER_TO_KEY.get(m, m) for m in models]

    # parse cosines
    cos_list = [float(x.strip()) for x in args.min_cosines.split(",") if x.strip()]

    # parse gap days → (label, days, seconds)
    gap_specs: List[Tuple[str, Optional[int], Optional[int]]] = []
    seen: Set[str] = set()
    for raw in args.max_pair_gap_days.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok.lower() in {"all", "any", "none", "null", "inf"}:
            if "all" not in seen:
                gap_specs.append(("all", None, None))
                seen.add("all")
            continue
        try:
            d = int(float(tok))
        except Exception:
            raise SystemExit(f"Invalid token in --max-pair-gap-days: '{tok}'")
        if d <= 0:
            if "all" not in seen:
                gap_specs.append(("all", None, None))
                seen.add("all")
            continue
        key = f"d{d}"
        if key not in seen:
            gap_specs.append((f"<= {d}d", d, d * 24 * 3600))
            seen.add(key)

    if not gap_specs:
        raise SystemExit("--max-pair-gap-days produced no valid gap setting.")

    # load params
    if rank == 0:
        txt = args.params_by_model_json.strip()
        params_by_model = load_json(txt) if os.path.exists(txt) else json.loads(txt)
        if not isinstance(params_by_model, dict):
            raise SystemExit("--params-by-model-json must be a dict mapping model→dict")
    else:
        params_by_model = None
    params_by_model = comm.bcast(params_by_model, root=0)

    # distribute files
    if rank == 0:
        train_files = list_json_files(args.train_dir) if "train" in splits else []
        test_files = list_json_files(args.test_dir) if "test" in splits else []
        print(f"[rank0] train_files={len(train_files)} test_files={len(test_files)}", flush=True)
    else:
        train_files = test_files = None
    train_files = comm.bcast(train_files, root=0)
    test_files = comm.bcast(test_files, root=0)

    my_train = [train_files[i] for i in range(len(train_files)) if i % size == rank]
    my_test = [test_files[i] for i in range(len(test_files)) if i % size == rank]

    # load topics (union of all needed IDs)
    needed_local = collect_needed_book_ids(my_train) | collect_needed_book_ids(my_test)
    needed_all = comm.gather(needed_local, root=0)
    if rank == 0:
        needed_union: Set[str] = set()
        for s in needed_all:
            needed_union |= set(s)
        needed_list = sorted(needed_union)
        print(f"[rank0] needed_book_ids={len(needed_list)}", flush=True)
    else:
        needed_list = None
    needed_list = comm.bcast(needed_list, root=0)

    book_to_topic = load_book_topics_from_npz(args.book_topics_npz, set(needed_list))
    if rank == 0:
        print(f"[rank0] loaded_topics={len(book_to_topic)}", flush=True)
    comm.Barrier()

    summary: Dict[str, Any] = {
        "status": "ok",
        "world_size": int(size),
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "splits": splits,
        "book_topics_npz": args.book_topics_npz,
        "min_rating_diff": float(args.min_rating_diff),
        "warmup": int(args.warmup),
        "min_cosines": cos_list,
        "max_pair_gap_days": [g[1] for g in gap_specs],
        "models": models,
        "per_user_csv_dir": str(per_user_dir),
        "runs": [],
    }

    for cos in cos_list:
        cos_key = f"{cos:.2f}"

        for gap_label, gap_days, gap_seconds in gap_specs:
            if rank == 0:
                label = "ALL VALID PAIRS" if gap_days is None else f"Δt{gap_label}"
                print(f"\n\n=== COMBO: min_cosine={cos_key} AND {label} ===", flush=True)
            comm.Barrier()

            for model in models:
                mp = dict(params_by_model.get(model, {}))
                mp.pop("min_cosine", None)

                if rank == 0:
                    gap_text = "all" if gap_days is None else str(gap_days)
                    print(f"\n[rank0] model={model} min_cosine={cos_key} max_gap_days={gap_text}", flush=True)
                comm.Barrier()

                train_stats = test_stats = None

                if "train" in splits:
                    train_stats = run_full_eval_split_mpi(
                        comm=comm, my_files=my_train, model_key=model,
                        model_params=mp, min_rating_diff=float(args.min_rating_diff),
                        min_cosine=float(cos), max_pair_time_gap_seconds=gap_seconds,
                        book_to_topic=book_to_topic, mse_mode="train_same_timestep",
                        warmup=int(args.warmup),
                    )

                if "test" in splits:
                    test_stats = run_full_eval_split_mpi(
                        comm=comm, my_files=my_test, model_key=model,
                        model_params=mp, min_rating_diff=float(args.min_rating_diff),
                        min_cosine=float(cos), max_pair_time_gap_seconds=gap_seconds,
                        book_to_topic=book_to_topic, mse_mode="test_one_step_ahead",
                        warmup=int(args.warmup),
                    )

                if rank == 0:
                    gap_tag = "all" if gap_days is None else f"{gap_days}d"
                    title_suffix = "ALL" if gap_days is None else f"<= {gap_days}d"
                    per_user_csv_info: Dict[str, str] = {}
                    run_record: Dict[str, Any] = {
                        "model": model,
                        "min_cosine": float(cos),
                        "max_pair_gap_label": gap_label,
                        "max_pair_gap_days": int(gap_days) if gap_days is not None else None,
                        "max_pair_time_gap_seconds": int(gap_seconds) if gap_seconds is not None else None,
                        "fixed_model_hyperparameters": mp,
                    }

                    if train_stats:
                        print_block(f"{model} TRAIN [{cos_key}, {title_suffix}]", train_stats)
                        train_csv = per_user_dir / f"per_user__{model}__cos{cos_key}__gap{gap_tag}__train.csv"
                        write_per_user_csv(str(train_csv), train_stats.get("per_user_rows", []))
                        print(f"[rank0] wrote {train_csv}", flush=True)
                        run_record["train"] = {k: v for k, v in train_stats.items() if k != "per_user_rows"}
                        per_user_csv_info["train"] = str(train_csv)

                    if test_stats:
                        print_block(f"{model} TEST  [{cos_key}, {title_suffix}]", test_stats)
                        test_csv = per_user_dir / f"per_user__{model}__cos{cos_key}__gap{gap_tag}__test.csv"
                        write_per_user_csv(str(test_csv), test_stats.get("per_user_rows", []))
                        print(f"[rank0] wrote {test_csv}", flush=True)
                        run_record["test"] = {k: v for k, v in test_stats.items() if k != "per_user_rows"}
                        per_user_csv_info["test"] = str(test_csv)

                    run_record["per_user_csv"] = per_user_csv_info
                    summary["runs"].append(run_record)

                comm.Barrier()

    if rank != 0:
        return

    out_json = str(out_dir / "GLOBAL__all_models_summary.json")
    save_json(out_json, summary)
    print(f"\n[rank0] Saved summary: {out_json}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rank_label = "rank?"
        try:
            rank_label = f"rank{MPI.COMM_WORLD.Get_rank()}"
        except Exception:
            pass
        print(f"[{rank_label}] FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        try:
            MPI.COMM_WORLD.Abort(1)
        except Exception:
            raise
