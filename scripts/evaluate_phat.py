#!/usr/bin/env python3
"""
Evaluate p-hat-driven generated users for Bayesian preference-change detection models.

Assumed self-directed log semantics:

  X[t]      = generated_topic_vector          (ground-truth consumed item theta_t)
  y[t]      = rating                          (ground-truth rating r_t)
  labels[t] = preference_change_label
  P[t]      = preference_vector               (ground-truth preference vector p_t)

For model M, data_dir = <self_root>/<M>. Legacy roots that still contain
<self_root>/<M>/all are also supported.

Preference-change detection:
- The score is continuous KL divergence between posterior at t and previous posterior.
- Metrics: ROC and PR over timesteps >= warmup, reported as percentages.

RATING EVALUATION (TEST ONLY, prequential / one-step-ahead):
- Do NOT use model.predict().
- For timestep t>0:
    rhat_t = mu_{t-1}^T x_t          (mean BEFORE updating on t, i.e., after updating on t-1)
    accumulate (rhat_t - r_t)^2
    then update on (x_t, r_t)
- Warmup applies to evaluation only (default warmup=10):
    evaluate MAE/MSE only for t >= warmup.

Preference tracking (post-update):
- PD_t   = ||P_true_t - P_hat_t||_2
- RelPE_t = scalar aggregated relative preference error:
    mean_coord: mean_i |Δ_i| / max(|P_i|, eps)
    l1_ratio:   sum_i |Δ_i| / max(sum_i |P_i|, eps)
    l2_ratio:   ||Δ||_2 / max(||P||_2, eps)

Tracking lag / recovery:
For each preference-change event at timestep t0 >= warmup:
  lag = first k>=0 such that metric[t0+k] <= threshold
  search stops at (next event - 1). If not recovered before that, it's "not recovered".

Reported for each threshold:
- TR  (Tracking Ratio): recovered_events / total_events
- Lag: mean lag over recovered events only

Combined capped metric:
For each event e:
  T_e = L_e if recovered
  T_e = cap if not recovered   (cap defaults to 10, typically cycle length)
Then:
  MTL = (1/N) * sum_e T_e
"""

import os
import glob
import json
import argparse
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from sklearn.metrics import roc_auc_score, average_precision_score

from nspb.hyperparameters import load_hyperparameter_arg, load_hyperparameter_group
from nspb.posterior_distances import kl_divergence, nig_kl_divergence
from nspb.models import (
    BayesianModel,
    VarianceBoundedBayesianModel,
    BayesianForgettingFactorModel,
    AROW_Regression,
    KalmanFilter,
    BayesianSlidingWindowModel,
    PowerPriorBayesianModel,
    NormalInverseGammaModel,  # NIG
)

# =========================
# Defaults
# =========================
ALL_MODELS = ["KF", "BLR", "vbBLR", "fBLR", "AROW", "BLRsw", "PBLR", "NIG"]

DEFAULT_WARMUP_STEPS = 10

DEFAULT_PD_THRESHOLDS: List[float] = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
DEFAULT_RELP_THRESHOLDS: List[float] = [0.25]

DEFAULT_RELP_EPS = 1e-8
DEFAULT_RELP_MODE = "mean_coord"  # mean_coord | l1_ratio | l2_ratio

DEFAULT_CAP_LAG = 10  # cycle length in your synthetic dataset

MODEL_DISPLAY = {
    "KF":    "KF-AF",
    "AROW":  "AROW",
    "BLR":   "BLR",
    "vbBLR": "BLR-VB",
    "fBLR":  "BLR-FF",
    "BLRsw": "BLR-SW",
    "PBLR":  "BLR-PP",
    "NIG":   "BLR-NIG",
}

PAPER_TO_KEY = {v: k for k, v in MODEL_DISPLAY.items()}

DEFAULT_PARAMS_BY_MODEL: Dict[str, Dict[str, float]] = load_hyperparameter_group("evaluation")

# =========================
# Data loading for self-directed logs
# =========================
def load_user_file_self_directed(
    path: str,
    x_key: str = "generated_topic_vector",
    y_key: str = "rating",
    label_key: str = "preference_change_label",
    pref_key: str = "preference_vector",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds:
      X[t]      = generated_topic_vector
      y[t]      = rating
      labels[t] = preference_change_label
      P[t]      = preference_vector
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Normalize structures
    if isinstance(data, str):
        lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
        data = [json.loads(ln) for ln in lines]

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]
        elif "records" in data and isinstance(data["records"], list):
            data = data["records"]
        elif all(isinstance(k, str) and k.isdigit() for k in data.keys()):
            items = sorted(((int(k), v) for k, v in data.items()), key=lambda kv: kv[0])
            data = [v for _, v in items]
        else:
            raise ValueError(f"Unrecognized JSON dict structure in {path}. Keys: {list(data.keys())[:20]}")

    if not isinstance(data, list) or (len(data) > 0 and not isinstance(data[0], dict)):
        raise ValueError(f"Expected list of dict records in {path}, got {type(data)}")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    labels_list: List[int] = []
    P_list: List[np.ndarray] = []

    for rec in data:
        t = rec.get("timestep", "?")

        x_val = rec.get(x_key, None)
        if x_val is None:
            raise ValueError(f"{path}: missing {x_key} at timestep {t}")

        y_val = rec.get(y_key, None)
        if y_val is None:
            raise ValueError(f"{path}: missing {y_key} at timestep {t}")

        lab = rec.get(label_key, None)
        if lab is None:
            raise ValueError(f"{path}: missing {label_key} at timestep {t}")

        p = rec.get(pref_key, None)
        if p is None:
            raise ValueError(f"{path}: missing {pref_key} at timestep {t}")

        X_list.append(np.asarray(x_val, dtype=np.float64))
        y_list.append(float(y_val))
        labels_list.append(int(lab))
        P_list.append(np.asarray(p, dtype=np.float64))

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.float64)
    labels = np.asarray(labels_list, dtype=int)
    P = np.vstack(P_list)

    return X, y, labels, P


def list_user_paths_with_folder(data_dir: str) -> List[Tuple[str, str]]:
    """
    Return list of (path, folder_name). For top-level files, folder_name='ROOT'.
    Looks for *.json directly under data_dir and under each immediate subfolder.
    """
    results: List[Tuple[str, str]] = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        results.append((p, "ROOT"))
    for p in sorted(glob.glob(os.path.join(data_dir, "*", "*.json"))):
        folder = os.path.basename(os.path.dirname(p))
        results.append((p, folder))
    return results


# =========================
# Models
# =========================
def build_model(model_key: str, n_topics: int, params: Dict[str, float]):
    if model_key == "KF":
        return KalmanFilter(
            n_topics,
            params.get("variance_p", 3.0),
            params.get("variance",   0.3),
            params.get("delta",      0.015),
            params.get("eta",        1e-5),
        )
    elif model_key == "BLR":
        return BayesianModel(
            n_topics,
            params.get("default_variance", 50.0),
            params.get("noise_precision",  0.33),
        )
    elif model_key == "vbBLR":
        return VarianceBoundedBayesianModel(
            n_topics,
            params.get("default_variance", 15.0),
            params.get("noise_precision",  0.33),
            params.get("tau",              1.0),
        )
    elif model_key == "fBLR":
        return BayesianForgettingFactorModel(
            n_topics,
            params.get("default_variance", 50.0),
            params.get("noise_precision",  0.33),
        )
    elif model_key == "AROW":
        return AROW_Regression(
            n_topics,
            params.get("lam1", 0.8),
            params.get("lam2", 1.0),
        )
    elif model_key == "BLRsw":
        return BayesianSlidingWindowModel(
            n_topics,
            params.get("m", 25),
            params.get("default_variance", 50.0),
            params.get("noise_precision",  0.33),
        )
    elif model_key == "PBLR":
        return PowerPriorBayesianModel(
            n_topics,
            params.get("alpha", 0.5),
            params.get("default_variance", 50.0),
            params.get("noise_precision",  0.33),
        )
    elif model_key == "NIG":
        return NormalInverseGammaModel(
            n_topics,
            params.get("default_variance", 50.0),
            params.get("default_a", 2.0),
            params.get("default_b", 2.0),
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def _mean_vector_from_model(model_key: str, model) -> np.ndarray:
    """
    Extract the current mean preference estimate vector (pre-update).
    This is the mean used to predict the NEXT observation (prequential).
    """
    if model_key == "NIG":
        p, _, _, _ = model.get_params()
        return np.asarray(p, dtype=np.float64).ravel()

    mu, *_ = model.get_params()
    return np.asarray(mu, dtype=np.float64).ravel()


# =========================
# Relative preference error: |P - Phat| / |P|
# =========================
def relp_scalar(P_true: np.ndarray, P_hat: np.ndarray, eps: float, mode: str) -> float:
    P_true = np.asarray(P_true, dtype=np.float64).ravel()
    P_hat = np.asarray(P_hat, dtype=np.float64).ravel()
    if P_true.shape != P_hat.shape:
        return float("nan")

    delta = np.abs(P_true - P_hat)

    if mode == "mean_coord":
        denom = np.maximum(np.abs(P_true), eps)
        v = delta / denom
        return float(np.mean(v))

    if mode == "l1_ratio":
        denom = max(float(np.sum(np.abs(P_true))), float(eps))
        return float(np.sum(delta) / denom)

    if mode == "l2_ratio":
        denom = max(float(np.linalg.norm(P_true)), float(eps))
        return float(np.linalg.norm(P_true - P_hat) / denom)

    raise ValueError(f"Unknown relp mode: {mode}")


# =========================
# Traces (KL, PD, RelPE, TEST-prequential rating errors)
# =========================
def compute_traces(
    model_key: str,
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, float],
    P_true: np.ndarray,
    relp_eps: float,
    relp_mode: str,
) -> Dict[str, np.ndarray]:
    """
    TEST-only / prequential rating evaluation:
      - For t>0: rhat_t = mu_{t-1}^T x_t (mean BEFORE updating on t)
      - store abs error + squared error in err_test[t], err2_test[t]
      - then update on (x_t, r_t)
    Warmup is applied later in user_metrics (evaluate only t>=warmup).
    """
    T, n_topics = X.shape

    kl = np.zeros(T, dtype=np.float64)
    pd = np.zeros(T, dtype=np.float64)
    relp = np.zeros(T, dtype=np.float64)

    # TEST-prequential errors
    err_test = np.full(T, np.nan, dtype=np.float64)
    err2_test = np.full(T, np.nan, dtype=np.float64)

    # ---------- NIG ----------
    if model_key == "NIG":
        model = build_model(model_key, n_topics, params)
        prev_p, prev_V, prev_a, prev_b = model.get_params()

        for t in range(T):
            x_t = X[t]
            r_t = float(y[t])

            # prequential prediction (before update on t)
            if t > 0:
                p_prev = _mean_vector_from_model(model_key, model)
                rhat = float(np.dot(p_prev, x_t))
                e = r_t - rhat
                err_test[t] = abs(e)
                err2_test[t] = e * e

            # update on t
            p_hat_t, V_t, a_t, b_t = model.update(x_t, r_t)
            p_hat_vec = np.asarray(p_hat_t, dtype=np.float64).ravel()

            # tracking (post-update)
            P_t = P_true[t].ravel()
            pd[t] = float(np.linalg.norm(P_t - p_hat_vec))
            relp[t] = relp_scalar(P_t, p_hat_vec, eps=relp_eps, mode=relp_mode)

            # KL(post_t || post_{t-1})
            kl_t = nig_kl_divergence(prev_V, V_t, prev_p, p_hat_t, prev_a, a_t, prev_b, b_t)
            if not np.isfinite(kl_t) or kl_t < 0:
                kl_t = 0.0 if not np.isfinite(kl_t) else max(0.0, float(kl_t))
            kl[t] = float(kl_t)

            prev_p, prev_V, prev_a, prev_b = p_hat_t, V_t, a_t, b_t

        return {"kl": kl, "pd": pd, "relp": relp, "err_test": err_test, "err2_test": err2_test}

    # ---------- Other models ----------
    model = build_model(model_key, n_topics, params)
    prev_mu, prev_S = model.get_params()
    if model_key == "AROW":
        prev_mu = np.asarray(prev_mu).ravel()

    for t in range(T):
        x_t = X[t]
        r_t = float(y[t])

        # prequential prediction (before update on t)
        if t > 0:
            mu_prev = _mean_vector_from_model(model_key, model)
            rhat = float(np.dot(mu_prev, x_t))
            e = r_t - rhat
            err_test[t] = abs(e)
            err2_test[t] = e * e

        # update on t
        if model_key == "vbBLR":
            out = model.update(x_t, r_t)
            if isinstance(out, tuple) and len(out) == 3:
                mu_t, _, S_t = out
            else:
                mu_t, S_t = out
        elif model_key == "fBLR":
            mu_t, S_t = model.update(x_t, r_t, rho=params.get("rho", 0.98))
        else:
            mu_t, S_t = model.update(x_t, r_t)

        mu_t = np.asarray(mu_t, dtype=np.float64)
        S_t = np.asarray(S_t, dtype=np.float64)
        if model_key == "AROW":
            mu_t = mu_t.ravel()

        # tracking (post-update)
        P_t = P_true[t].ravel()
        p_hat_vec = mu_t.ravel()
        pd[t] = float(np.linalg.norm(P_t - p_hat_vec))
        relp[t] = relp_scalar(P_t, p_hat_vec, eps=relp_eps, mode=relp_mode)

        # KL(post_t || post_{t-1})
        kl_t = kl_divergence(S_t, prev_S, mu_t, prev_mu)
        if not np.isfinite(kl_t) or kl_t < 0:
            kl_t = 0.0 if not np.isfinite(kl_t) else max(0.0, float(kl_t))
        kl[t] = float(kl_t)

        prev_mu, prev_S = mu_t, S_t

    return {"kl": kl, "pd": pd, "relp": relp, "err_test": err_test, "err2_test": err2_test}


# =========================
# Tracking lag
# =========================
def tracking_lags_for_user(
    labels: np.ndarray,
    metric: np.ndarray,
    warmup_steps: int,
    thresholds: List[float],
    max_lag_horizon: Optional[int] = None,
) -> Dict[float, Tuple[List[int], int]]:
    """
    Returns per threshold:
      (lags_list, total_events)
    where lags_list contains only recovered event lags.
    """
    T = len(labels)
    event_times = [t for t in range(T) if labels[t] == 1 and t >= warmup_steps]
    out: Dict[float, Tuple[List[int], int]] = {th: ([], 0) for th in thresholds}
    if not event_times:
        return out

    for i, t0 in enumerate(event_times):
        t1 = event_times[i + 1] if i + 1 < len(event_times) else None
        end = (t1 - 1) if t1 is not None else (T - 1)
        if max_lag_horizon is not None and max_lag_horizon >= 0:
            end = min(end, t0 + int(max_lag_horizon))

        for th in thresholds:
            lags_list, total = out[th]
            total += 1

            for t in range(t0, end + 1):
                if np.isfinite(metric[t]) and metric[t] <= th:
                    lags_list.append(t - t0)
                    break

            out[th] = (lags_list, total)

    return out


# =========================
# Aggregation helpers
# =========================
def folder_group(folder_name: str) -> str:
    f = folder_name.lower()
    if f == "theta" or f.endswith("_theta"):
        return "theta"
    if f == "p" or f.endswith("_p"):
        return "p"
    return ""


def folder_ps_pb_family(folder_name: str) -> str:
    """
    PS/PB families over all variants:
      - ps_family: any folder starting with 'ps' (ps, ps_pc, ps_pc_multi, ...)
      - pb_family: any folder starting with 'pb' (pb, pb_pc, pb_pc_multi, ...)
    """
    f = folder_name.lower()
    if f.startswith("ps"):
        return "ps_family"
    if f.startswith("pb"):
        return "pb_family"
    return ""


def folder_requested_aggregate(folder_name: str) -> str:
    """
    Requested theta+p-combined buckets:
      - ps_only    : ps_* folders that are NOT ps_pc*
      - pb_only    : pb_* folders that are NOT pb_pc*
      - ps_plus_pc : all ps_pc* folders
      - pb_plus_pc : all pb_pc* folders
    """
    f = folder_name.lower()

    if f.startswith("ps_") and not f.startswith("ps_pc_"):
        return "ps_only"
    if f.startswith("pb_") and not f.startswith("pb_pc_"):
        return "pb_only"
    if f.startswith("ps_pc_"):
        return "ps_plus_pc"
    if f.startswith("pb_pc_"):
        return "pb_plus_pc"
    return ""


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    try:
        roc = float(roc_auc_score(y_true, scores))
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, scores))
    except Exception:
        pr = float("nan")
    return roc, pr


def nanmean_list(x: List[float]) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def fmt(x: float, nd: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def fmt_pct(x: float, nd: int = 1) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{100.0 * x:.{nd}f}"


def fmt_lag(x: float, nd: int = 1) -> str:
    if np.isposinf(x):
        return "∞"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


# =========================
# Per-user metrics (TEST only)
# =========================
def user_metrics(traces: Dict[str, np.ndarray], labels: np.ndarray, warmup: int) -> Dict[str, float]:
    T = len(labels)
    eval_mask_auc = np.arange(T) >= warmup
    if eval_mask_auc.sum() == 0:
        return {}

    roc, pr = safe_auc(labels[eval_mask_auc], traces["kl"][eval_mask_auc])

    idx = np.arange(T)
    eval_mask_err = (idx >= warmup) & np.isfinite(traces["err_test"])
    mae = float(np.mean(traces["err_test"][eval_mask_err])) if eval_mask_err.sum() else float("nan")
    mse = float(np.mean(traces["err2_test"][eval_mask_err])) if eval_mask_err.sum() else float("nan")

    pd_mean = float(np.mean(traces["pd"][eval_mask_auc]))
    relp_mean = float(np.mean(traces["relp"][eval_mask_auc]))
    return {"roc_auc": roc, "pr_auc": pr, "mae": mae, "mse": mse, "pd": pd_mean, "relp": relp_mean}


# =========================
# Paper tracking metric helpers
# =========================
def capped_mtl_from_lags(lags: List[int], total_events: int, cap: int) -> float:
    """
    Paper MTL metric:
      T_e = L_e if recovered else cap
      MTL = (sum(L_e) + (N - recovered)*cap) / N
    """
    if total_events <= 0:
        return float("nan")
    rec = len(lags)
    s = float(np.sum(lags)) if rec else 0.0
    return float((s + float(total_events - rec) * float(cap)) / float(total_events))


def paper_threshold(thresholds: List[float]) -> float:
    return 0.25 if 0.25 in thresholds else thresholds[0]


def paper_tracking_values(
    lag_dict: Dict[float, Dict[str, Any]],
    threshold: float,
    cap_lag: int,
) -> Tuple[float, float, float, int, int]:
    lags = lag_dict[threshold]["lags"]
    total = int(lag_dict[threshold]["total"])
    recovered = len(lags)
    tr = (recovered / total) if total else float("nan")
    lag = float(np.mean(lags)) if recovered else (float("inf") if total else float("nan"))
    mtl = capped_mtl_from_lags(lags, total, cap_lag)
    return tr, lag, mtl, recovered, total


def paper_metric_line(
    metrics: Dict[str, float],
    lag_dict: Dict[float, Dict[str, Any]],
    threshold: float,
    cap_lag: int,
) -> str:
    tr, lag, mtl, _, _ = paper_tracking_values(lag_dict, threshold, cap_lag)
    return (
        f"MSE={fmt(metrics['mse'], 3)} | "
        f"ROC={fmt_pct(metrics['roc_auc'])} | "
        f"PR={fmt_pct(metrics['pr_auc'])} | "
        f"TR/Lag={fmt_pct(tr)}/{fmt_lag(lag)} | "
        f"MTL={fmt_lag(mtl)}"
    )


# =========================
# Evaluate one model on its self-directed model folder
# =========================
def evaluate_one_model(
    model_key: str,
    data_dir: str,
    warmup: int,
    pd_thresholds: List[float],
    relp_thresholds: List[float],
    params: Dict[str, float],
    relp_eps: float,
    relp_mode: str,
    cap_lag: int,
):
    user_items = list_user_paths_with_folder(data_dir)
    print("\n==============================")
    print(f"MODEL: {model_key}  |  Dir: {data_dir}")
    print(f"Users found: {len(user_items)}")
    print(f"Warmup: {warmup} (evaluate from timestep {warmup + 1})")
    print(f"PD thresholds: {pd_thresholds}")
    print(f"RelPE thresholds: {relp_thresholds} | RelPE mode: {relp_mode} | eps={relp_eps}")
    print("Rating error: TEST-prequential (one-step-ahead), dot-product only (no predict())")
    print(f"MTL cap: {cap_lag}")
    print("==============================\n")

    used = 0
    skipped = 0

    roc_list: List[float] = []
    pr_list: List[float] = []
    mae_list: List[float] = []
    mse_list: List[float] = []
    pd_list: List[float] = []
    relp_list: List[float] = []

    folder_metrics: Dict[str, Dict[str, List[float]]] = {}
    group_metrics: Dict[str, Dict[str, List[float]]] = {"theta": {}, "p": {}}
    family_metrics: Dict[str, Dict[str, List[float]]] = {"ps_family": {}, "pb_family": {}}
    requested_agg_metrics: Dict[str, Dict[str, List[float]]] = {
        "ps_only": {},
        "pb_only": {},
        "ps_plus_pc": {},
        "pb_plus_pc": {},
    }

    def init_bucket():
        return {"roc_auc": [], "pr_auc": [], "mae": [], "mse": [], "pd": [], "relp": []}

    # lag aggregators
    pd_lag_overall = {th: {"lags": [], "total": 0} for th in pd_thresholds}
    relp_lag_overall = {th: {"lags": [], "total": 0} for th in relp_thresholds}

    pd_lag_folder: Dict[str, Dict[float, Dict[str, Any]]] = {}
    relp_lag_folder: Dict[str, Dict[float, Dict[str, Any]]] = {}

    pd_lag_group = {
        "theta": {th: {"lags": [], "total": 0} for th in pd_thresholds},
        "p": {th: {"lags": [], "total": 0} for th in pd_thresholds},
    }
    relp_lag_group = {
        "theta": {th: {"lags": [], "total": 0} for th in relp_thresholds},
        "p": {th: {"lags": [], "total": 0} for th in relp_thresholds},
    }
    relp_lag_family = {
        "ps_family": {th: {"lags": [], "total": 0} for th in relp_thresholds},
        "pb_family": {th: {"lags": [], "total": 0} for th in relp_thresholds},
    }
    relp_lag_requested_agg = {
        "ps_only": {th: {"lags": [], "total": 0} for th in relp_thresholds},
        "pb_only": {th: {"lags": [], "total": 0} for th in relp_thresholds},
        "ps_plus_pc": {th: {"lags": [], "total": 0} for th in relp_thresholds},
        "pb_plus_pc": {th: {"lags": [], "total": 0} for th in relp_thresholds},
    }

    for path, folder in user_items:
        try:
            X, y, labels, P = load_user_file_self_directed(path)
        except Exception:
            skipped += 1
            continue

        T = X.shape[0]
        if T <= warmup:
            skipped += 1
            continue

        g = folder_group(folder)
        fam = folder_ps_pb_family(folder)
        requested_agg = folder_requested_aggregate(folder)

        traces = compute_traces(
            model_key, X, y, params, P,
            relp_eps=relp_eps, relp_mode=relp_mode
        )

        met = user_metrics(traces, labels, warmup)
        if not met:
            skipped += 1
            continue

        used += 1
        roc_list.append(met["roc_auc"])
        pr_list.append(met["pr_auc"])
        mae_list.append(met["mae"])
        mse_list.append(met["mse"])
        pd_list.append(met["pd"])
        relp_list.append(met["relp"])

        # folder metrics
        if folder not in folder_metrics:
            folder_metrics[folder] = init_bucket()
        for k in folder_metrics[folder]:
            folder_metrics[folder][k].append(met[k])

        # group metrics
        if g in ("theta", "p"):
            if not group_metrics[g]:
                group_metrics[g] = init_bucket()
            for k in group_metrics[g]:
                group_metrics[g][k].append(met[k])

        # PS/PB family metrics across all theta+p users
        if fam in ("ps_family", "pb_family"):
            if not family_metrics[fam]:
                family_metrics[fam] = init_bucket()
            for k in family_metrics[fam]:
                family_metrics[fam][k].append(met[k])

        # Requested aggregates (theta+p combined)
        if requested_agg in requested_agg_metrics:
            if not requested_agg_metrics[requested_agg]:
                requested_agg_metrics[requested_agg] = init_bucket()
            for k in requested_agg_metrics[requested_agg]:
                requested_agg_metrics[requested_agg][k].append(met[k])

        # tracking lags: PD
        user_pd_lags = tracking_lags_for_user(
            labels,
            traces["pd"],
            warmup,
            pd_thresholds,
            max_lag_horizon=cap_lag,
        )
        for th in pd_thresholds:
            lags, total = user_pd_lags[th]
            pd_lag_overall[th]["lags"].extend(lags)
            pd_lag_overall[th]["total"] += total

            if folder not in pd_lag_folder:
                pd_lag_folder[folder] = {th2: {"lags": [], "total": 0} for th2 in pd_thresholds}
            pd_lag_folder[folder][th]["lags"].extend(lags)
            pd_lag_folder[folder][th]["total"] += total

            if g in ("theta", "p"):
                pd_lag_group[g][th]["lags"].extend(lags)
                pd_lag_group[g][th]["total"] += total

        # tracking lags: RelPE
        user_relp_lags = tracking_lags_for_user(
            labels,
            traces["relp"],
            warmup,
            relp_thresholds,
            max_lag_horizon=cap_lag,
        )
        for th in relp_thresholds:
            lags, total = user_relp_lags[th]
            relp_lag_overall[th]["lags"].extend(lags)
            relp_lag_overall[th]["total"] += total

            if folder not in relp_lag_folder:
                relp_lag_folder[folder] = {th2: {"lags": [], "total": 0} for th2 in relp_thresholds}
            relp_lag_folder[folder][th]["lags"].extend(lags)
            relp_lag_folder[folder][th]["total"] += total

            if g in ("theta", "p"):
                relp_lag_group[g][th]["lags"].extend(lags)
                relp_lag_group[g][th]["total"] += total
            if fam in ("ps_family", "pb_family"):
                relp_lag_family[fam][th]["lags"].extend(lags)
                relp_lag_family[fam][th]["total"] += total
            if requested_agg in relp_lag_requested_agg:
                relp_lag_requested_agg[requested_agg][th]["lags"].extend(lags)
                relp_lag_requested_agg[requested_agg][th]["total"] += total

    paper_recovery_threshold = paper_threshold(relp_thresholds)

    def print_tracking_block(lag_dict: Dict[float, Dict[str, Any]], thresholds: List[float], cap: int):
        for th in thresholds:
            tr, lag, mtl, recovered, total = paper_tracking_values(lag_dict, th, cap)
            print(
                f"    RelPE≤{th:0.2f}: "
                f"TR={fmt_pct(tr)} | "
                f"Lag={fmt_lag(lag)} | "
                f"MTL={fmt_lag(mtl)} | "
                f"Events={recovered}/{total}"
            )

    # ===== Overall =====
    roc = nanmean_list(roc_list)
    pr  = nanmean_list(pr_list)
    mae = nanmean_list(mae_list)
    mse = nanmean_list(mse_list)
    pdm = nanmean_list(pd_list)
    relpm = nanmean_list(relp_list)

    print(f"=== Overall (macro over users; from timestep {warmup + 1}) ===")
    print(f"{MODEL_DISPLAY[model_key]} ({model_key}):")
    print(f"  Users used: {used} | skipped: {skipped}")
    summary = {"roc_auc": roc, "pr_auc": pr, "mse": mse}
    print(f"  Paper metrics: {paper_metric_line(summary, relp_lag_overall, paper_recovery_threshold, cap_lag)}")
    print(f"  Diagnostics: MAE={fmt(mae, 6)} | PD={fmt(pdm, 6)} | RelPE={fmt(relpm, 6)}")
    print(f"  Preference recovery:")
    print(f"  RelPE={relp_mode}")
    print_tracking_block(relp_lag_overall, relp_thresholds, cap_lag)
    print()

    # ===== Per-folder =====
    print(f"=== Per-folder (macro over users in folder) ===")
    folders = sorted(folder_metrics.keys())
    if not folders:
        print("  (no evaluable users)")
    else:
        for folder in folders:
            roc_f = nanmean_list(folder_metrics[folder]["roc_auc"])
            pr_f  = nanmean_list(folder_metrics[folder]["pr_auc"])
            mae_f = nanmean_list(folder_metrics[folder]["mae"])
            mse_f = nanmean_list(folder_metrics[folder]["mse"])
            pd_f  = nanmean_list(folder_metrics[folder]["pd"])
            relp_f = nanmean_list(folder_metrics[folder]["relp"])
            summary = {"roc_auc": roc_f, "pr_auc": pr_f, "mse": mse_f}
            print(f"  - {folder}: {paper_metric_line(summary, relp_lag_folder.get(folder, {th: {'lags': [], 'total': 0} for th in relp_thresholds}), paper_recovery_threshold, cap_lag)}")
            print(f"    Diagnostics: MAE={fmt(mae_f, 6)} | PD={fmt(pd_f, 6)} | RelPE={fmt(relp_f, 6)}")
            print(f"    Preference recovery:")
            print_tracking_block(
                relp_lag_folder.get(folder, {th: {"lags": [], "total": 0} for th in relp_thresholds}),
                relp_thresholds,
                cap_lag
            )
    print()

    # ===== THETA vs P =====
    print(f"=== THETA vs P (macro over users) ===")
    for g in ("theta", "p"):
        if not group_metrics[g]:
            roc_g = pr_g = mae_g = mse_g = pd_g = relp_g = float("nan")
            n_users = 0
        else:
            roc_g = nanmean_list(group_metrics[g]["roc_auc"])
            pr_g  = nanmean_list(group_metrics[g]["pr_auc"])
            mae_g = nanmean_list(group_metrics[g]["mae"])
            mse_g = nanmean_list(group_metrics[g]["mse"])
            pd_g  = nanmean_list(group_metrics[g]["pd"])
            relp_g = nanmean_list(group_metrics[g]["relp"])
            n_users = len(group_metrics[g]["roc_auc"])

        summary = {"roc_auc": roc_g, "pr_auc": pr_g, "mse": mse_g}
        print(f"  {g.upper():9s}: {paper_metric_line(summary, relp_lag_group[g], paper_recovery_threshold, cap_lag)} (users: {n_users})")
        print(f"    Diagnostics: MAE={fmt(mae_g, 6)} | PD={fmt(pd_g, 6)} | RelPE={fmt(relp_g, 6)}")
        print(f"    Preference recovery:")
        print_tracking_block(relp_lag_group[g], relp_thresholds, cap_lag)
    print()

    # ===== PS/PB Family (all variants) =====
    print(f"=== PS/PB Family Averages (all ps/pb variants; macro over users) ===")
    print("No theta/p split in this section (all combined).")
    print("PS family = all ps* folders (ps + ps_pc + ps_pc_multi + ...)")
    print("PB family = all pb* folders (pb + pb_pc + pb_pc_multi + ...)")
    for fam, fam_label in (("ps_family", "PS_FAMILY"), ("pb_family", "PB_FAMILY")):
        if not family_metrics[fam]:
            roc_f = pr_f = mae_f = mse_f = pd_f = relp_f = float("nan")
            n_users = 0
        else:
            roc_f = nanmean_list(family_metrics[fam]["roc_auc"])
            pr_f = nanmean_list(family_metrics[fam]["pr_auc"])
            mae_f = nanmean_list(family_metrics[fam]["mae"])
            mse_f = nanmean_list(family_metrics[fam]["mse"])
            pd_f = nanmean_list(family_metrics[fam]["pd"])
            relp_f = nanmean_list(family_metrics[fam]["relp"])
            n_users = len(family_metrics[fam]["roc_auc"])

        summary = {"roc_auc": roc_f, "pr_auc": pr_f, "mse": mse_f}
        print(
            f"  {fam_label:10s}: "
            f"{paper_metric_line(summary, relp_lag_family[fam], paper_recovery_threshold, cap_lag)} "
            f"(users: {n_users})"
        )
        print(f"    Diagnostics: MAE={fmt(mae_f, 6)} | PD={fmt(pd_f, 6)} | RelPE={fmt(relp_f, 6)}")
        print("    Preference recovery:")
        print_tracking_block(relp_lag_family[fam], relp_thresholds, cap_lag)
    print()

    # ===== Requested aggregates =====
    print("=== Requested Aggregates (theta + p combined) ===")
    print("PS_ONLY    = ps_* folders excluding ps_pc*")
    print("PB_ONLY    = pb_* folders excluding pb_pc*")
    print("PS_PLUS_PC = all ps_pc* folders")
    print("PB_PLUS_PC = all pb_pc* folders")
    for key, label in (
        ("ps_only", "PS_ONLY"),
        ("pb_only", "PB_ONLY"),
        ("ps_plus_pc", "PS_PLUS_PC"),
        ("pb_plus_pc", "PB_PLUS_PC"),
    ):
        if not requested_agg_metrics[key]:
            roc_k = pr_k = mae_k = mse_k = pd_k = relp_k = float("nan")
            n_users = 0
        else:
            roc_k = nanmean_list(requested_agg_metrics[key]["roc_auc"])
            pr_k = nanmean_list(requested_agg_metrics[key]["pr_auc"])
            mae_k = nanmean_list(requested_agg_metrics[key]["mae"])
            mse_k = nanmean_list(requested_agg_metrics[key]["mse"])
            pd_k = nanmean_list(requested_agg_metrics[key]["pd"])
            relp_k = nanmean_list(requested_agg_metrics[key]["relp"])
            n_users = len(requested_agg_metrics[key]["roc_auc"])

        summary = {"roc_auc": roc_k, "pr_auc": pr_k, "mse": mse_k}
        print(
            f"  {label:10s}: "
            f"{paper_metric_line(summary, relp_lag_requested_agg[key], paper_recovery_threshold, cap_lag)} "
            f"(users: {n_users})"
        )
        print(f"    Diagnostics: MAE={fmt(mae_k, 6)} | PD={fmt(pd_k, 6)} | RelPE={fmt(relp_k, 6)}")
        print("    Preference recovery:")
        print_tracking_block(relp_lag_requested_agg[key], relp_thresholds, cap_lag)
    print()


# =========================
# CLI helpers
# =========================
def parse_thresholds(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    parts = s.replace(",", " ").split()
    return [float(p) for p in parts]


def load_params_by_model(arg: Optional[str]) -> Dict[str, Dict[str, float]]:
    """Load evaluation hyperparameters from default config, JSON file, or JSON string."""
    return load_hyperparameter_arg(arg, "evaluation")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate self-directed generated users in <self_root>/<MODEL>.")
    p.add_argument("--self-root", required=True, help="Root directory containing per-model folders (KF, BLR, ...)")
    p.add_argument("--models", "-m", nargs="+", default=["all"], help="Models to run. Use 'all' or keys (KF BLR ...).")
    p.add_argument("--list-models", "-l", action="store_true", help="List available models and exit.")
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Warmup steps; evaluate from warmup+1 onward.")

    p.add_argument("--pd-thresholds", type=str, default="", help="PD thresholds, e.g. '0.05,0.1,0.2,0.25,...'")
    p.add_argument("--relp-thresholds", type=str, default="", help="RelPE thresholds for tracking lag. The paper uses 0.25.")

    p.add_argument("--relp-eps", type=float, default=DEFAULT_RELP_EPS, help="Epsilon for RelPE denominator.")
    p.add_argument("--relp-mode", type=str, default=DEFAULT_RELP_MODE, choices=["mean_coord", "l1_ratio", "l2_ratio"],
                   help="Aggregate |P-P_hat|/|P| into a scalar.")

    p.add_argument("--cap-lag", type=int, default=DEFAULT_CAP_LAG,
                   help="Penalty assigned to non-recovered events when computing paper MTL.")

    p.add_argument("--params-by-model-json", type=str, default=None, help="JSON string or file path for model hyperparameters.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("Available models:")
        for k in ALL_MODELS:
            print(f"  {k:6s} - {MODEL_DISPLAY[k]}")
        return

    if "all" in [m.lower() for m in args.models]:
        models_to_run = ALL_MODELS
    else:
        models_to_run = []
        for m in args.models:
            mk = PAPER_TO_KEY.get(m, m)  # accept paper names (KF-AF, BLR-VB, …)
            if mk in ALL_MODELS:
                models_to_run.append(mk)
            else:
                print(f"Warning: Unknown model '{m}' (skipping).")
        if not models_to_run:
            raise SystemExit(f"No valid models selected. Options: {', '.join(MODEL_DISPLAY.values())}")

    pd_thresholds = parse_thresholds(args.pd_thresholds) if args.pd_thresholds.strip() else DEFAULT_PD_THRESHOLDS
    relp_thresholds = parse_thresholds(args.relp_thresholds) if args.relp_thresholds.strip() else DEFAULT_RELP_THRESHOLDS

    params_by_model = load_params_by_model(args.params_by_model_json)
    for m in models_to_run:
        if m not in params_by_model:
            raise SystemExit(f"params_by_model_json is missing parameters for model '{m}'.")

    self_root = args.self_root

    print("==============================================")
    print("SELF-DIRECTED EVALUATION (MODEL-MATCHED)")
    print("RATING: TEST-prequential (one-step-ahead), dot-product only (no predict())")
    print(f"Self root       : {self_root}")
    print(f"Warmup steps    : {args.warmup_steps}")
    print(f"PD thresholds   : {pd_thresholds}")
    print(f"RelPE thresholds: {relp_thresholds}")
    print(f"RelPE mode      : {args.relp_mode} | eps={args.relp_eps}")
    print(f"MTL cap         : {args.cap_lag}")
    print(f"Models          : {', '.join(models_to_run)}")
    print("==============================================\n")

    for m in models_to_run:
        model_root = os.path.join(self_root, m)
        legacy_all_dir = os.path.join(model_root, "all")
        data_dir = legacy_all_dir if os.path.isdir(legacy_all_dir) else model_root
        if not os.path.isdir(data_dir):
            print(f"[WARN] Missing directory for model {m}: {data_dir} (skipping)")
            continue

        evaluate_one_model(
            model_key=m,
            data_dir=data_dir,
            warmup=args.warmup_steps,
            pd_thresholds=pd_thresholds,
            relp_thresholds=relp_thresholds,
            params=params_by_model[m],
            relp_eps=args.relp_eps,
            relp_mode=args.relp_mode,
            cap_lag=args.cap_lag,
        )


if __name__ == "__main__":
    main()
