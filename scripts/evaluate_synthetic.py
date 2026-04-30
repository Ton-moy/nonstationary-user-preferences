"""
Evaluation script for Bayesian preference-change detection models.

Adds Relative Preference Error based on vectors:
    RelPE_t = |P_t - P_hat_t| / |P_t|   (vector)
Since tracking lag needs a scalar per timestep, we aggregate this vector into ONE scalar:

Default (recommended): mean over coordinates
    relp_t = mean_i ( |P_t[i] - P_hat_t[i]| / max(|P_t[i]|, eps) )

(You can also choose L1-ratio or L2-ratio via --relp-mode.)

Other features:
- Evaluates TRAIN and TEST sets separately (two directories).
- Preference-change scores use continuous KL divergence values:
    - ROC (roc_auc_score, reported as percent)
    - PR (average_precision_score, reported as percent)
- Rating error metrics:
    - TRAIN (after-update): error at t uses prediction after updating on (x_t, r_t)
    - TEST  (before-update-nextstep): prediction for (x_t, r_t) is made using model state after processing (t-1)
- Preference tracking:
    - PD_t = ||P_t - P_hat_t||_2  (L2 distance)
    - RelPE_t (scalar aggregated from vector relative errors, see above)
    - Tracking Lag for multiple thresholds (computed for both PD and RelPE):
        For each ground-truth preference change event (label=1):
            lag = min k>=0 s.t. metric[t0+k] <= threshold
            search stops at next event - 1 (if a new change occurs before recovery -> not recovered)
        Report:
            - TR = recovered / total events
            - Lag = mean lag over recovered events only
            - MTL = mean tracking lag over all events, with cap penalty for non-recovered events
- No per-user printing; only aggregated results:
    - Overall (macro across users)
    - Per-folder
    - THETA vs P groups
    - Family aggregates:
        - PS family: (ps + ps_pc + ps_pc_multi)
        - PB family: (pb + pb_pc + pb_pc_multi)
      each reported separately for THETA and P

Usage:
    python evaluate.py --models all --train-data-dir ... --test-data-dir ...
"""

import os
import glob
import json
import argparse
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from sklearn.metrics import roc_auc_score, average_precision_score

# Uses your existing utilities/classes
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
# DEFAULT CONFIG (overridable via CLI)
# =========================
DEFAULT_TRAIN_DATA_DIR = "/path/to/train"
DEFAULT_TEST_DATA_DIR = "/path/to/test"
DEFAULT_WARMUP_STEPS = 10

ALL_MODELS = ["KF", "BLR", "vbBLR", "fBLR", "AROW", "BLRsw", "PBLR", "NIG"]

DEFAULT_PD_THRESHOLDS: List[float] = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
DEFAULT_RELP_THRESHOLDS: List[float] = [0.25]

DEFAULT_RELP_EPS = 1e-8
# relp-mode:
#   mean_coord: mean_i |Δ_i|/max(|P_i|,eps)
#   l1_ratio:   sum_i |Δ_i| / max(sum_i |P_i|, eps)
#   l2_ratio:   ||Δ||_2 / max(||P||_2, eps)
DEFAULT_RELP_MODE = "mean_coord"
DEFAULT_CAP_LAG = 10

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

DEFAULT_PARAMS_BY_MODEL: Dict[str, Dict[str, float]] = load_hyperparameter_group("evaluation")


# =========================
# Data loading
# =========================
def load_user_file(
    path: str,
    x_key: str = "topic_vector",
    y_key: str = "rating",
    label_key: str = "preference_change_label",
    pref_key: str = "preference_vector",
):
    with open(path, "r") as f:
        data = json.load(f)

    # Robust normalization: allow list-of-dicts, wrapped dicts, dict indexed by timestep, or JSONL string
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

    X = np.vstack([np.asarray(rec[x_key], dtype=np.float64) for rec in data])
    y = np.asarray([float(rec[y_key]) for rec in data], dtype=np.float64)
    labels = np.asarray([int(rec[label_key]) for rec in data], dtype=int)
    P = np.vstack([np.asarray(rec[pref_key], dtype=np.float64) for rec in data])
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


def _safe_float_pred(pred) -> float:
    arr = np.asarray(pred).ravel()
    return float(arr[0])


# =========================
# Relative preference error: |P - Phat| / |P|
# =========================
def relp_scalar(P_true: np.ndarray, P_hat: np.ndarray, eps: float, mode: str) -> float:
    """
    Compute scalar RelPE_t from vector expression |P_t - P_hat_t| / |P_t|.

    mode:
      - mean_coord: mean_i |Δ_i|/max(|P_i|,eps)    (closest to your literal expression coordinate-wise)
      - l1_ratio:   sum_i |Δ_i| / max(sum_i |P_i|, eps)
      - l2_ratio:   ||Δ||_2 / max(||P||_2, eps)

    Returns float (may be nan if inputs invalid).
    """
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
# Traces (KL, PD, RelPE, rating errors)
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
    Returns per-timestep traces:
      - kl: (T,) KL(post_t || post_{t-1})  [computed after update at t]
      - pd: (T,) ||P_true_t - P_hat_t||_2  [after update at t]
      - relp: (T,) scalar RelPE_t from |P - Phat|/|P|
      - err_after, err2_after: (T,)   after-update rating errors at t
      - err_next,  err2_next:  (T,)   next-step rating errors aligned to target index t (nan at t=0)
    """
    T, n_topics = X.shape

    kl = np.zeros(T, dtype=np.float64)
    pd = np.zeros(T, dtype=np.float64)
    relp = np.zeros(T, dtype=np.float64)

    err_after = np.zeros(T, dtype=np.float64)
    err2_after = np.zeros(T, dtype=np.float64)

    err_next = np.full(T, np.nan, dtype=np.float64)
    err2_next = np.full(T, np.nan, dtype=np.float64)

    # NIG
    if model_key == "NIG":
        model = build_model(model_key, n_topics, params)
        prev_p, prev_V, prev_a, prev_b = model.get_params()

        for t in range(T):
            x_t = X[t]
            r_t = float(y[t])

            # next-step prediction for current t using state BEFORE update at t
            if t > 0:
                rhat_next = _safe_float_pred(model.predict(x_t))
                e_next = abs(r_t - rhat_next)
                err_next[t] = e_next
                err2_next[t] = e_next * e_next

            # update on (x_t, r_t)
            p_hat_t, V_t, a_t, b_t = model.update(x_t, r_t)
            p_hat_vec = np.asarray(p_hat_t, dtype=np.float64).ravel()

            # after-update prediction on x_t
            rhat_after = _safe_float_pred(model.predict(x_t))
            e = abs(r_t - rhat_after)
            err_after[t] = e
            err2_after[t] = e * e

            P_t = P_true[t].ravel()
            pd[t] = float(np.linalg.norm(P_t - p_hat_vec))
            relp[t] = relp_scalar(P_t, p_hat_vec, eps=relp_eps, mode=relp_mode)

            # KL
            kl_t = nig_kl_divergence(prev_V, V_t, prev_p, p_hat_t, prev_a, a_t, prev_b, b_t)
            if not np.isfinite(kl_t) or kl_t < 0:
                kl_t = 0.0 if not np.isfinite(kl_t) else max(0.0, float(kl_t))
            kl[t] = float(kl_t)

            prev_p, prev_V, prev_a, prev_b = p_hat_t, V_t, a_t, b_t

        return {
            "kl": kl,
            "pd": pd,
            "relp": relp,
            "err_after": err_after,
            "err2_after": err2_after,
            "err_next": err_next,
            "err2_next": err2_next,
        }

    # Other models
    model = build_model(model_key, n_topics, params)
    prev_mu, prev_S = model.get_params()
    if model_key == "AROW":
        prev_mu = np.asarray(prev_mu).ravel()

    for t in range(T):
        x_t = X[t]
        r_t = float(y[t])

        # next-step style prediction for current t using state BEFORE update at t
        if t > 0:
            rhat_next = _safe_float_pred(model.predict(x_t))
            e_next = abs(r_t - rhat_next)
            err_next[t] = e_next
            err2_next[t] = e_next * e_next

        # update
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

        # after-update prediction
        rhat_after = _safe_float_pred(model.predict(x_t))
        e = abs(r_t - rhat_after)
        err_after[t] = e
        err2_after[t] = e * e

        P_t = P_true[t].ravel()
        p_hat_vec = mu_t.ravel()
        pd[t] = float(np.linalg.norm(P_t - p_hat_vec))
        relp[t] = relp_scalar(P_t, p_hat_vec, eps=relp_eps, mode=relp_mode)

        # KL(post || prev)
        kl_t = kl_divergence(S_t, prev_S, mu_t, prev_mu)
        if not np.isfinite(kl_t) or kl_t < 0:
            kl_t = 0.0 if not np.isfinite(kl_t) else max(0.0, float(kl_t))
        kl[t] = float(kl_t)

        prev_mu, prev_S = mu_t, S_t

    return {
        "kl": kl,
        "pd": pd,
        "relp": relp,
        "err_after": err_after,
        "err2_after": err2_after,
        "err_next": err_next,
        "err2_next": err2_next,
    }


# =========================
# Tracking lag
# =========================
def tracking_lags_for_user(
    labels: np.ndarray,
    metric: np.ndarray,
    warmup_steps: int,
    thresholds: List[float],
) -> Dict[float, Tuple[List[int], int]]:
    """
    For one user, compute tracking lags per threshold on a given metric sequence.

    Events: labels[t0] == 1 and t0 >= warmup_steps

    For each event at t0:
      window ends at next_event-1 (if any), else T-1.
      find first t in [t0, end] where metric[t] <= threshold
      lag = t - t0 (recovered)
      if none found => not recovered for that event

    Returns: threshold -> (lags_list, total_events)
             avg lag computed on recovered-only; recovery rate = len(lags_list)/total_events
    """
    T = len(labels)
    event_times = [t for t in range(T) if labels[t] == 1 and t >= warmup_steps]
    out: Dict[float, Tuple[List[int], int]] = {th: ([], 0) for th in thresholds}
    if not event_times:
        return out

    for i, t0 in enumerate(event_times):
        t1 = event_times[i + 1] if i + 1 < len(event_times) else None
        end = (t1 - 1) if t1 is not None else (T - 1)

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


def folder_family(folder_name: str) -> str:
    """
    Map folder to coarse families requested by analysis:
      - ps_family: folders related to ps / ps_pc / ps_pc_multi
      - pb_family: folders related to pb / pb_pc / pb_pc_multi
    """
    f = folder_name.lower()
    if "ps" in f:
        return "ps_family"
    if "pb" in f:
        return "pb_family"
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
# Split evaluation
# =========================
def user_metrics_train(traces: Dict[str, np.ndarray], labels: np.ndarray, warmup: int) -> Dict[str, float]:
    T = len(labels)
    eval_mask = np.arange(T) >= warmup
    if eval_mask.sum() == 0:
        return {}
    roc, pr = safe_auc(labels[eval_mask], traces["kl"][eval_mask])
    mae = float(np.mean(traces["err_after"][eval_mask]))
    mse = float(np.mean(traces["err2_after"][eval_mask]))
    pd_mean = float(np.mean(traces["pd"][eval_mask]))
    relp_mean = float(np.mean(traces["relp"][eval_mask]))
    return {"roc_auc": roc, "pr_auc": pr, "mae": mae, "mse": mse, "pd": pd_mean, "relp": relp_mean}


def user_metrics_test(traces: Dict[str, np.ndarray], labels: np.ndarray, warmup: int) -> Dict[str, float]:
    T = len(labels)
    eval_mask_auc = np.arange(T) >= warmup
    if eval_mask_auc.sum() == 0:
        return {}
    roc, pr = safe_auc(labels[eval_mask_auc], traces["kl"][eval_mask_auc])

    # next-step errors exist for t>=1, and we also apply warmup on the target index t
    idx = np.arange(T)
    eval_mask_err = (idx >= warmup) & np.isfinite(traces["err_next"])
    mae = float(np.mean(traces["err_next"][eval_mask_err])) if eval_mask_err.sum() else float("nan")
    mse = float(np.mean(traces["err2_next"][eval_mask_err])) if eval_mask_err.sum() else float("nan")

    pd_mean = float(np.mean(traces["pd"][eval_mask_auc]))
    relp_mean = float(np.mean(traces["relp"][eval_mask_auc]))
    return {"roc_auc": roc, "pr_auc": pr, "mae": mae, "mse": mse, "pd": pd_mean, "relp": relp_mean}


def evaluate_split(
    data_dir: str,
    models_to_run: List[str],
    split_name: str,
    warmup: int,
    pd_thresholds: List[float],
    relp_thresholds: List[float],
    params_by_model: Dict[str, Dict[str, float]],
    relp_eps: float,
    relp_mode: str,
    cap_lag: int,
):
    user_items = list_user_paths_with_folder(data_dir)
    if not user_items:
        print(f"[{split_name}] No JSON files found in {data_dir}")
        return

    print("\n==============================")
    print(f"{split_name} EVALUATION")
    print(f"Data dir: {data_dir}")
    print(f"Users found: {len(user_items)}")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Warmup: {warmup} (evaluate from timestep {warmup + 1})")
    print(f"PD thresholds: {pd_thresholds}")
    print(f"RelPE thresholds: {relp_thresholds} | RelPE mode: {relp_mode} | eps={relp_eps}")
    print(f"MTL cap: {cap_lag}")
    print("==============================\n")

    used = {m: 0 for m in models_to_run}
    skipped = {m: 0 for m in models_to_run}

    roc_list = {m: [] for m in models_to_run}
    pr_list  = {m: [] for m in models_to_run}
    mae_list = {m: [] for m in models_to_run}
    mse_list = {m: [] for m in models_to_run}
    pd_list  = {m: [] for m in models_to_run}
    relp_list= {m: [] for m in models_to_run}

    folder_metrics: Dict[str, Dict[str, Dict[str, List[float]]]] = {m: {} for m in models_to_run}
    group_metrics: Dict[str, Dict[str, Dict[str, List[float]]]] = {m: {"theta": {}, "p": {}} for m in models_to_run}
    family_metrics: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {
        m: {
            "ps_family": {"theta": {}, "p": {}},
            "pb_family": {"theta": {}, "p": {}},
        }
        for m in models_to_run
    }

    def init_metric_bucket() -> Dict[str, List[float]]:
        return {"roc_auc": [], "pr_auc": [], "mae": [], "mse": [], "pd": [], "relp": []}

    # event-level lag aggregators for PD and RelPE
    pd_lag_overall = {m: {th: {"lags": [], "total": 0} for th in pd_thresholds} for m in models_to_run}
    pd_lag_folder  = {m: {} for m in models_to_run}
    pd_lag_group   = {
        m: {
            "theta": {th: {"lags": [], "total": 0} for th in pd_thresholds},
            "p": {th: {"lags": [], "total": 0} for th in pd_thresholds},
        }
        for m in models_to_run
    }
    pd_lag_family = {
        m: {
            "ps_family": {
                "theta": {th: {"lags": [], "total": 0} for th in pd_thresholds},
                "p": {th: {"lags": [], "total": 0} for th in pd_thresholds},
            },
            "pb_family": {
                "theta": {th: {"lags": [], "total": 0} for th in pd_thresholds},
                "p": {th: {"lags": [], "total": 0} for th in pd_thresholds},
            },
        }
        for m in models_to_run
    }

    relp_lag_overall = {m: {th: {"lags": [], "total": 0} for th in relp_thresholds} for m in models_to_run}
    relp_lag_folder  = {m: {} for m in models_to_run}
    relp_lag_group   = {
        m: {
            "theta": {th: {"lags": [], "total": 0} for th in relp_thresholds},
            "p": {th: {"lags": [], "total": 0} for th in relp_thresholds},
        }
        for m in models_to_run
    }
    relp_lag_family = {
        m: {
            "ps_family": {
                "theta": {th: {"lags": [], "total": 0} for th in relp_thresholds},
                "p": {th: {"lags": [], "total": 0} for th in relp_thresholds},
            },
            "pb_family": {
                "theta": {th: {"lags": [], "total": 0} for th in relp_thresholds},
                "p": {th: {"lags": [], "total": 0} for th in relp_thresholds},
            },
        }
        for m in models_to_run
    }

    for path, folder in user_items:
        try:
            X, y, labels, P = load_user_file(path)
        except Exception:
            for m in models_to_run:
                skipped[m] += 1
            continue

        T = X.shape[0]
        if T <= warmup:
            for m in models_to_run:
                skipped[m] += 1
            continue

        g = folder_group(folder)

        for m in models_to_run:
            if m not in params_by_model:
                raise ValueError(f"params_by_model missing key: {m}")

            traces = compute_traces(
                m, X, y, params_by_model[m], P,
                relp_eps=relp_eps, relp_mode=relp_mode
            )

            if split_name.upper() == "TRAIN":
                met = user_metrics_train(traces, labels, warmup)
            else:
                met = user_metrics_test(traces, labels, warmup)

            if not met:
                skipped[m] += 1
                continue

            used[m] += 1
            roc_list[m].append(met["roc_auc"])
            pr_list[m].append(met["pr_auc"])
            mae_list[m].append(met["mae"])
            mse_list[m].append(met["mse"])
            pd_list[m].append(met["pd"])
            relp_list[m].append(met["relp"])

            # folder metrics
            if folder not in folder_metrics[m]:
                folder_metrics[m][folder] = init_metric_bucket()
            for k in folder_metrics[m][folder]:
                folder_metrics[m][folder][k].append(met[k])

            # group metrics
            if g in ("theta", "p"):
                if not group_metrics[m][g]:
                    group_metrics[m][g] = init_metric_bucket()
                for k in group_metrics[m][g]:
                    group_metrics[m][g][k].append(met[k])

            # requested family metrics (PS/PB) split by theta/p
            fam = folder_family(folder)
            if fam in ("ps_family", "pb_family") and g in ("theta", "p"):
                if not family_metrics[m][fam][g]:
                    family_metrics[m][fam][g] = init_metric_bucket()
                for k in family_metrics[m][fam][g]:
                    family_metrics[m][fam][g][k].append(met[k])

            # lag: PD
            user_pd_lags = tracking_lags_for_user(labels, traces["pd"], warmup, pd_thresholds)
            for th in pd_thresholds:
                lags, total = user_pd_lags[th]
                pd_lag_overall[m][th]["lags"].extend(lags)
                pd_lag_overall[m][th]["total"] += total

                if folder not in pd_lag_folder[m]:
                    pd_lag_folder[m][folder] = {th2: {"lags": [], "total": 0} for th2 in pd_thresholds}
                pd_lag_folder[m][folder][th]["lags"].extend(lags)
                pd_lag_folder[m][folder][th]["total"] += total

                if g in ("theta", "p"):
                    pd_lag_group[m][g][th]["lags"].extend(lags)
                    pd_lag_group[m][g][th]["total"] += total
                if fam in ("ps_family", "pb_family") and g in ("theta", "p"):
                    pd_lag_family[m][fam][g][th]["lags"].extend(lags)
                    pd_lag_family[m][fam][g][th]["total"] += total

            # lag: RelPE
            user_relp_lags = tracking_lags_for_user(labels, traces["relp"], warmup, relp_thresholds)
            for th in relp_thresholds:
                lags, total = user_relp_lags[th]
                relp_lag_overall[m][th]["lags"].extend(lags)
                relp_lag_overall[m][th]["total"] += total

                if folder not in relp_lag_folder[m]:
                    relp_lag_folder[m][folder] = {th2: {"lags": [], "total": 0} for th2 in relp_thresholds}
                relp_lag_folder[m][folder][th]["lags"].extend(lags)
                relp_lag_folder[m][folder][th]["total"] += total

                if g in ("theta", "p"):
                    relp_lag_group[m][g][th]["lags"].extend(lags)
                    relp_lag_group[m][g][th]["total"] += total
                if fam in ("ps_family", "pb_family") and g in ("theta", "p"):
                    relp_lag_family[m][fam][g][th]["lags"].extend(lags)
                    relp_lag_family[m][fam][g][th]["total"] += total

    paper_recovery_threshold = paper_threshold(relp_thresholds)

    def print_tracking_block(lag_dict: Dict[float, Dict[str, Any]], thresholds: List[float], label: str):
        print(f"  {label}:")
        for th in thresholds:
            tr, lag, mtl, recovered, total = paper_tracking_values(lag_dict, th, cap_lag)
            print(
                f"    RelPE≤{th:0.2f}: "
                f"TR={fmt_pct(tr)} | "
                f"Lag={fmt_lag(lag)} | "
                f"MTL={fmt_lag(mtl)} | "
                f"Events={recovered}/{total}"
            )

    err_label = "rating diagnostics after update" if split_name.upper() == "TRAIN" else "rating diagnostics before next-step update"

    # ===== Overall =====
    print(f"=== [{split_name}] Overall (macro over users; from timestep {warmup + 1}) ===")
    for m in models_to_run:
        roc = nanmean_list(roc_list[m])
        pr  = nanmean_list(pr_list[m])
        mae = nanmean_list(mae_list[m])
        mse = nanmean_list(mse_list[m])
        pdm = nanmean_list(pd_list[m])
        relpm = nanmean_list(relp_list[m])

        print(f"{MODEL_DISPLAY[m]} ({m}):")
        print(f"  Users used: {used[m]} | skipped: {skipped[m]}")
        summary = {"roc_auc": roc, "pr_auc": pr, "mse": mse}
        print(f"  Paper metrics: {paper_metric_line(summary, relp_lag_overall[m], paper_recovery_threshold, cap_lag)}")
        print(f"  Diagnostics: MAE={fmt(mae, 6)} | PD={fmt(pdm, 6)} | RelPE={fmt(relpm, 6)} ({err_label})")
        print(f"  Preference recovery:")
        print_tracking_block(relp_lag_overall[m], relp_thresholds, label=f"RelPE={relp_mode}")
        print()

    # ===== Per-folder =====
    print(f"=== [{split_name}] Per-folder (macro over users in folder) ===")
    for m in models_to_run:
        folders = sorted(folder_metrics[m].keys())
        print(f"{MODEL_DISPLAY[m]} ({m}):")
        if not folders:
            print("  (no evaluable users)")
            continue
        for folder in folders:
            roc = nanmean_list(folder_metrics[m][folder]["roc_auc"])
            pr  = nanmean_list(folder_metrics[m][folder]["pr_auc"])
            mae = nanmean_list(folder_metrics[m][folder]["mae"])
            mse = nanmean_list(folder_metrics[m][folder]["mse"])
            pdm = nanmean_list(folder_metrics[m][folder]["pd"])
            relpm = nanmean_list(folder_metrics[m][folder]["relp"])
            summary = {"roc_auc": roc, "pr_auc": pr, "mse": mse}
            print(f"  - {folder}: {paper_metric_line(summary, relp_lag_folder[m][folder], paper_recovery_threshold, cap_lag)}")
            print(f"    Diagnostics: MAE={fmt(mae, 6)} | PD={fmt(pdm, 6)} | RelPE={fmt(relpm, 6)}")
            print(f"    Preference recovery:")
            print_tracking_block(relp_lag_folder[m][folder], relp_thresholds, label="RelPE")
        print()

    # ===== THETA vs P =====
    print(f"=== [{split_name}] THETA vs P (macro over users) ===")
    for m in models_to_run:
        print(f"{MODEL_DISPLAY[m]} ({m}):")
        for g in ("theta", "p"):
            if not group_metrics[m][g]:
                roc = pr = mae = mse = pdm = relpm = float("nan")
                n_users = 0
            else:
                roc = nanmean_list(group_metrics[m][g]["roc_auc"])
                pr  = nanmean_list(group_metrics[m][g]["pr_auc"])
                mae = nanmean_list(group_metrics[m][g]["mae"])
                mse = nanmean_list(group_metrics[m][g]["mse"])
                pdm = nanmean_list(group_metrics[m][g]["pd"])
                relpm = nanmean_list(group_metrics[m][g]["relp"])
                n_users = len(group_metrics[m][g]["roc_auc"])

            summary = {"roc_auc": roc, "pr_auc": pr, "mse": mse}
            print(f"  {g.upper():9s}: {paper_metric_line(summary, relp_lag_group[m][g], paper_recovery_threshold, cap_lag)} (users: {n_users})")
            print(f"    Diagnostics: MAE={fmt(mae, 6)} | PD={fmt(pdm, 6)} | RelPE={fmt(relpm, 6)}")
            print(f"    Preference recovery:")
            print_tracking_block(relp_lag_group[m][g], relp_thresholds, label="RelPE")
        print()

    # ===== Requested family averages =====
    print(f"=== [{split_name}] PS/PB Family Averages (macro over users) ===")
    print("PS family = ps + ps_pc + ps_pc_multi")
    print("PB family = pb + pb_pc + pb_pc_multi")
    for m in models_to_run:
        print(f"{MODEL_DISPLAY[m]} ({m}):")
        for fam, fam_label in (("ps_family", "PS_FAMILY"), ("pb_family", "PB_FAMILY")):
            for g in ("theta", "p"):
                if not family_metrics[m][fam][g]:
                    roc = pr = mae = mse = pdm = relpm = float("nan")
                    n_users = 0
                else:
                    roc = nanmean_list(family_metrics[m][fam][g]["roc_auc"])
                    pr  = nanmean_list(family_metrics[m][fam][g]["pr_auc"])
                    mae = nanmean_list(family_metrics[m][fam][g]["mae"])
                    mse = nanmean_list(family_metrics[m][fam][g]["mse"])
                    pdm = nanmean_list(family_metrics[m][fam][g]["pd"])
                    relpm = nanmean_list(family_metrics[m][fam][g]["relp"])
                    n_users = len(family_metrics[m][fam][g]["roc_auc"])

                summary = {"roc_auc": roc, "pr_auc": pr, "mse": mse}
                print(
                    f"  {fam_label:10s} {g.upper():9s}: "
                    f"{paper_metric_line(summary, relp_lag_family[m][fam][g], paper_recovery_threshold, cap_lag)} "
                    f"(users: {n_users})"
                )
                print(f"    Diagnostics: MAE={fmt(mae, 6)} | PD={fmt(pdm, 6)} | RelPE={fmt(relpm, 6)}")
                print(f"    Preference recovery:")
                print_tracking_block(relp_lag_family[m][fam][g], relp_thresholds, label="RelPE")
        print()


# =========================
# CLI parsing (including params + thresholds)
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
    parser = argparse.ArgumentParser(
        description="Evaluate Bayesian preference-change detection models with paper metrics: MSE, ROC, PR, TR/Lag, and MTL."
    )
    parser.add_argument("--models", "-m", nargs="+", default=["all"],
                        help="Models to run. Use 'all' or specify keys (KF BLR ...).")
    parser.add_argument("--list-models", "-l", action="store_true",
                        help="List available models and exit.")
    parser.add_argument("--train-data-dir", default=DEFAULT_TRAIN_DATA_DIR,
                        help="Train data directory.")
    parser.add_argument("--test-data-dir", default=DEFAULT_TEST_DATA_DIR,
                        help="Test data directory.")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS,
                        help="Warmup steps; evaluate from warmup+1 onward.")

    parser.add_argument("--pd-thresholds", type=str, default="",
                        help="PD thresholds for tracking lag. Example: '0.05,0.1,0.2'")
    parser.add_argument("--relp-thresholds", type=str, default="",
                        help="RelPE thresholds for tracking lag. The paper uses 0.25.")

    parser.add_argument("--relp-eps", type=float, default=DEFAULT_RELP_EPS,
                        help="Epsilon for RelPE denominator.")
    parser.add_argument("--relp-mode", type=str, default=DEFAULT_RELP_MODE,
                        choices=["mean_coord", "l1_ratio", "l2_ratio"],
                        help="How to aggregate vector |P-Phat|/|P| into a scalar.")
    parser.add_argument("--cap-lag", type=int, default=DEFAULT_CAP_LAG,
                        help="Penalty assigned to non-recovered events when computing paper MTL.")

    parser.add_argument("--params-by-model-json", type=str, default=None,
                        help="JSON string or file path for model hyperparameters.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("Available models:")
        for k in ALL_MODELS:
            print(f"  {k:6s} - {MODEL_DISPLAY[k]}")
        return

    # models selection
    if "all" in [m.lower() for m in args.models]:
        models_to_run = ALL_MODELS
    else:
        models_to_run = []
        for m in args.models:
            if m in ALL_MODELS:
                models_to_run.append(m)
            else:
                print(f"Warning: Unknown model '{m}' (skipping).")
        if not models_to_run:
            raise SystemExit(f"No valid models selected. Options: {', '.join(ALL_MODELS)}")

    pd_thresholds = parse_thresholds(args.pd_thresholds) if args.pd_thresholds.strip() else DEFAULT_PD_THRESHOLDS
    relp_thresholds = parse_thresholds(args.relp_thresholds) if args.relp_thresholds.strip() else DEFAULT_RELP_THRESHOLDS

    params_by_model = load_params_by_model(args.params_by_model_json)
    for m in models_to_run:
        if m not in params_by_model:
            raise SystemExit(f"params_by_model_json is missing parameters for model '{m}'.")

    # TRAIN
    evaluate_split(
        data_dir=args.train_data_dir,
        models_to_run=models_to_run,
        split_name="TRAIN",
        warmup=args.warmup_steps,
        pd_thresholds=pd_thresholds,
        relp_thresholds=relp_thresholds,
        params_by_model=params_by_model,
        relp_eps=args.relp_eps,
        relp_mode=args.relp_mode,
        cap_lag=args.cap_lag,
    )

    # TEST
    evaluate_split(
        data_dir=args.test_data_dir,
        models_to_run=models_to_run,
        split_name="TEST",
        warmup=args.warmup_steps,
        pd_thresholds=pd_thresholds,
        relp_thresholds=relp_thresholds,
        params_by_model=params_by_model,
        relp_eps=args.relp_eps,
        relp_mode=args.relp_mode,
        cap_lag=args.cap_lag,
    )


if __name__ == "__main__":
    main()
