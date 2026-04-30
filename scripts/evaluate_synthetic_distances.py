"""
Evaluation script for Bayesian preference-change detection models with dual scores.

This mirrors evaluate_synthetic.py but reports preference-change detection performance
using both:
  - KL divergence
  - Wasserstein distance

Rating, RelPE, and tracking-lag reporting use the paper names:
MSE, ROC, PR, TR/Lag, and MTL.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from evaluate_synthetic import (
    ALL_MODELS,
    DEFAULT_CAP_LAG,
    DEFAULT_PARAMS_BY_MODEL,
    DEFAULT_PD_THRESHOLDS,
    DEFAULT_RELP_EPS,
    DEFAULT_RELP_MODE,
    DEFAULT_RELP_THRESHOLDS,
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_TRAIN_DATA_DIR,
    DEFAULT_WARMUP_STEPS,
    MODEL_DISPLAY,
    _safe_float_pred,
    build_model,
    fmt,
    fmt_lag,
    fmt_pct,
    folder_family,
    folder_group,
    list_user_paths_with_folder,
    load_params_by_model,
    load_user_file,
    nanmean_list,
    parse_thresholds,
    paper_threshold,
    paper_tracking_values,
    relp_scalar,
    safe_auc,
    tracking_lags_for_user,
)
from nspb.posterior_distances import (
    kl_divergence,
    nig_kl_divergence,
    wasserstein_distance,
)


def _sanitize_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, float(value))


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
      - kl: (T,) KL(post_t || post_{t-1})
      - wasserstein: (T,) Wasserstein(post_t, post_{t-1})
      - pd: (T,) ||P_true_t - P_hat_t||_2
      - relp: (T,) scalar RelPE_t from |P - Phat|/|P|
      - err_after, err2_after: after-update rating errors
      - err_next, err2_next: before-update next-step errors
    """
    T, n_topics = X.shape

    kl = np.zeros(T, dtype=np.float64)
    wasser = np.zeros(T, dtype=np.float64)
    pd = np.zeros(T, dtype=np.float64)
    relp = np.zeros(T, dtype=np.float64)

    err_after = np.zeros(T, dtype=np.float64)
    err2_after = np.zeros(T, dtype=np.float64)

    err_next = np.full(T, np.nan, dtype=np.float64)
    err2_next = np.full(T, np.nan, dtype=np.float64)

    if model_key == "NIG":
        model = build_model(model_key, n_topics, params)
        prev_p, prev_V, prev_a, prev_b = model.get_params()

        for t in range(T):
            x_t = X[t]
            r_t = float(y[t])

            if t > 0:
                rhat_next = _safe_float_pred(model.predict(x_t))
                e_next = abs(r_t - rhat_next)
                err_next[t] = e_next
                err2_next[t] = e_next * e_next

            p_hat_t, V_t, a_t, b_t = model.update(x_t, r_t)
            p_hat_vec = np.asarray(p_hat_t, dtype=np.float64).ravel()

            rhat_after = _safe_float_pred(model.predict(x_t))
            e = abs(r_t - rhat_after)
            err_after[t] = e
            err2_after[t] = e * e

            P_t = P_true[t].ravel()
            pd[t] = float(np.linalg.norm(P_t - p_hat_vec))
            relp[t] = relp_scalar(P_t, p_hat_vec, eps=relp_eps, mode=relp_mode)

            kl_t = nig_kl_divergence(prev_V, V_t, prev_p, p_hat_t, prev_a, a_t, prev_b, b_t)
            wd_t = wasserstein_distance(V_t, prev_V, p_hat_t, prev_p)

            kl[t] = _sanitize_score(kl_t)
            wasser[t] = _sanitize_score(wd_t)

            prev_p, prev_V, prev_a, prev_b = p_hat_t, V_t, a_t, b_t

        return {
            "kl": kl,
            "wasserstein": wasser,
            "pd": pd,
            "relp": relp,
            "err_after": err_after,
            "err2_after": err2_after,
            "err_next": err_next,
            "err2_next": err2_next,
        }

    model = build_model(model_key, n_topics, params)
    prev_mu, prev_S = model.get_params()
    if model_key == "AROW":
        prev_mu = np.asarray(prev_mu).ravel()

    for t in range(T):
        x_t = X[t]
        r_t = float(y[t])

        if t > 0:
            rhat_next = _safe_float_pred(model.predict(x_t))
            e_next = abs(r_t - rhat_next)
            err_next[t] = e_next
            err2_next[t] = e_next * e_next

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

        rhat_after = _safe_float_pred(model.predict(x_t))
        e = abs(r_t - rhat_after)
        err_after[t] = e
        err2_after[t] = e * e

        P_t = P_true[t].ravel()
        p_hat_vec = mu_t.ravel()
        pd[t] = float(np.linalg.norm(P_t - p_hat_vec))
        relp[t] = relp_scalar(P_t, p_hat_vec, eps=relp_eps, mode=relp_mode)

        kl_t = kl_divergence(S_t, prev_S, mu_t, prev_mu)
        wd_t = wasserstein_distance(S_t, prev_S, mu_t, prev_mu)

        kl[t] = _sanitize_score(kl_t)
        wasser[t] = _sanitize_score(wd_t)

        prev_mu, prev_S = mu_t, S_t

    return {
        "kl": kl,
        "wasserstein": wasser,
        "pd": pd,
        "relp": relp,
        "err_after": err_after,
        "err2_after": err2_after,
        "err_next": err_next,
        "err2_next": err2_next,
    }


def user_metrics_train(traces: Dict[str, np.ndarray], labels: np.ndarray, warmup: int) -> Dict[str, float]:
    T = len(labels)
    eval_mask = np.arange(T) >= warmup
    if eval_mask.sum() == 0:
        return {}

    kl_roc, kl_pr = safe_auc(labels[eval_mask], traces["kl"][eval_mask])
    wd_roc, wd_pr = safe_auc(labels[eval_mask], traces["wasserstein"][eval_mask])

    return {
        "kl_roc_auc": kl_roc,
        "kl_pr_auc": kl_pr,
        "kl_mean": float(np.mean(traces["kl"][eval_mask])),
        "wd_roc_auc": wd_roc,
        "wd_pr_auc": wd_pr,
        "wd_mean": float(np.mean(traces["wasserstein"][eval_mask])),
        "mae": float(np.mean(traces["err_after"][eval_mask])),
        "mse": float(np.mean(traces["err2_after"][eval_mask])),
        "pd": float(np.mean(traces["pd"][eval_mask])),
        "relp": float(np.mean(traces["relp"][eval_mask])),
    }


def user_metrics_test(traces: Dict[str, np.ndarray], labels: np.ndarray, warmup: int) -> Dict[str, float]:
    T = len(labels)
    eval_mask_auc = np.arange(T) >= warmup
    if eval_mask_auc.sum() == 0:
        return {}

    kl_roc, kl_pr = safe_auc(labels[eval_mask_auc], traces["kl"][eval_mask_auc])
    wd_roc, wd_pr = safe_auc(labels[eval_mask_auc], traces["wasserstein"][eval_mask_auc])

    idx = np.arange(T)
    eval_mask_err = (idx >= warmup) & np.isfinite(traces["err_next"])

    mae = float(np.mean(traces["err_next"][eval_mask_err])) if eval_mask_err.sum() else float("nan")
    mse = float(np.mean(traces["err2_next"][eval_mask_err])) if eval_mask_err.sum() else float("nan")

    return {
        "kl_roc_auc": kl_roc,
        "kl_pr_auc": kl_pr,
        "kl_mean": float(np.mean(traces["kl"][eval_mask_auc])),
        "wd_roc_auc": wd_roc,
        "wd_pr_auc": wd_pr,
        "wd_mean": float(np.mean(traces["wasserstein"][eval_mask_auc])),
        "mae": mae,
        "mse": mse,
        "pd": float(np.mean(traces["pd"][eval_mask_auc])),
        "relp": float(np.mean(traces["relp"][eval_mask_auc])),
    }


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
) -> None:
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
    print("Preference-change metrics: KL divergence + Wasserstein distance")
    print(f"PD thresholds: {pd_thresholds}")
    print(f"RelPE thresholds: {relp_thresholds} | RelPE mode: {relp_mode} | eps={relp_eps}")
    print(f"MTL cap: {cap_lag}")
    print("==============================\n")

    used = {m: 0 for m in models_to_run}
    skipped = {m: 0 for m in models_to_run}

    metric_keys = [
        "kl_roc_auc", "kl_pr_auc", "kl_mean",
        "wd_roc_auc", "wd_pr_auc", "wd_mean",
        "mae", "mse", "pd", "relp",
    ]

    overall_metrics = {m: {k: [] for k in metric_keys} for m in models_to_run}
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
        return {k: [] for k in metric_keys}

    pd_lag_overall = {m: {th: {"lags": [], "total": 0} for th in pd_thresholds} for m in models_to_run}
    pd_lag_folder = {m: {} for m in models_to_run}
    pd_lag_group = {
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
    relp_lag_folder = {m: {} for m in models_to_run}
    relp_lag_group = {
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

        if X.shape[0] <= warmup:
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
            for key in metric_keys:
                overall_metrics[m][key].append(met[key])

            if folder not in folder_metrics[m]:
                folder_metrics[m][folder] = init_metric_bucket()
            for key in metric_keys:
                folder_metrics[m][folder][key].append(met[key])

            if g in ("theta", "p"):
                if not group_metrics[m][g]:
                    group_metrics[m][g] = init_metric_bucket()
                for key in metric_keys:
                    group_metrics[m][g][key].append(met[key])

            fam = folder_family(folder)
            if fam in ("ps_family", "pb_family") and g in ("theta", "p"):
                if not family_metrics[m][fam][g]:
                    family_metrics[m][fam][g] = init_metric_bucket()
                for key in metric_keys:
                    family_metrics[m][fam][g][key].append(met[key])

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

    def print_tracking_block(lag_dict: Dict[float, Dict[str, Any]], thresholds: List[float], label: str) -> None:
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

    def summarize(bucket: Dict[str, List[float]]) -> Dict[str, float]:
        return {k: nanmean_list(v) for k, v in bucket.items()}

    def print_metric_lines(
        summary: Dict[str, float],
        lag_dict: Dict[float, Dict[str, Any]],
        err_label: str,
    ) -> None:
        tr, lag, mtl, _, _ = paper_tracking_values(lag_dict, paper_recovery_threshold, cap_lag)
        print(
            f"  KL paper metrics: "
            f"MSE={fmt(summary['mse'], 3)} | "
            f"ROC={fmt_pct(summary['kl_roc_auc'])} | "
            f"PR={fmt_pct(summary['kl_pr_auc'])} | "
            f"TR/Lag={fmt_pct(tr)}/{fmt_lag(lag)} | "
            f"MTL={fmt_lag(mtl)}"
        )
        print(
            f"  Wasserstein comparison: "
            f"ROC={fmt_pct(summary['wd_roc_auc'])} | "
            f"PR={fmt_pct(summary['wd_pr_auc'])}"
        )
        print(
            f"  Diagnostics: MAE={fmt(summary['mae'], 6)} | "
            f"PD={fmt(summary['pd'], 6)} | RelPE={fmt(summary['relp'], 6)} | "
            f"mean_KL={fmt(summary['kl_mean'], 6)} | mean_Wasserstein={fmt(summary['wd_mean'], 6)} "
            f"({err_label})"
        )

    err_label = "rating diagnostics after update" if split_name.upper() == "TRAIN" else "rating diagnostics before next-step update"

    print(f"=== [{split_name}] Overall (macro over users; from timestep {warmup + 1}) ===")
    for m in models_to_run:
        summary = summarize(overall_metrics[m])
        print(f"{MODEL_DISPLAY[m]} ({m}):")
        print(f"  Users used: {used[m]} | skipped: {skipped[m]}")
        print_metric_lines(summary, relp_lag_overall[m], err_label)
        print("  Preference recovery:")
        print_tracking_block(relp_lag_overall[m], relp_thresholds, label=f"RelPE={relp_mode}")
        print()

    print(f"=== [{split_name}] Per-folder (macro over users in folder) ===")
    for m in models_to_run:
        folders = sorted(folder_metrics[m].keys())
        print(f"{MODEL_DISPLAY[m]} ({m}):")
        if not folders:
            print("  (no evaluable users)")
            continue
        for folder in folders:
            summary = summarize(folder_metrics[m][folder])
            print(f"  - {folder}:")
            print_metric_lines(summary, relp_lag_folder[m][folder], "diagnostics")
            print("    Preference recovery:")
            print_tracking_block(relp_lag_folder[m][folder], relp_thresholds, label="RelPE")
        print()

    print(f"=== [{split_name}] THETA vs P (macro over users) ===")
    for m in models_to_run:
        print(f"{MODEL_DISPLAY[m]} ({m}):")
        for g in ("theta", "p"):
            bucket = group_metrics[m][g]
            if not bucket:
                summary = {k: float("nan") for k in metric_keys}
                n_users = 0
            else:
                summary = summarize(bucket)
                n_users = len(bucket["kl_roc_auc"])
            print(f"  {g.upper():9s} (users: {n_users})")
            print_metric_lines(summary, relp_lag_group[m][g], "diagnostics")
            print("    Preference recovery:")
            print_tracking_block(relp_lag_group[m][g], relp_thresholds, label="RelPE")
        print()

    print(f"=== [{split_name}] PS/PB Family Averages (macro over users) ===")
    print("PS family = ps + ps_pc + ps_pc_multi")
    print("PB family = pb + pb_pc + pb_pc_multi")
    for m in models_to_run:
        print(f"{MODEL_DISPLAY[m]} ({m}):")
        for fam, fam_label in (("ps_family", "PS_FAMILY"), ("pb_family", "PB_FAMILY")):
            for g in ("theta", "p"):
                bucket = family_metrics[m][fam][g]
                if not bucket:
                    summary = {k: float("nan") for k in metric_keys}
                    n_users = 0
                else:
                    summary = summarize(bucket)
                    n_users = len(bucket["kl_roc_auc"])
                print(f"  {fam_label:10s} {g.upper():9s} (users: {n_users})")
                print_metric_lines(summary, relp_lag_family[m][fam][g], "diagnostics")
                print("    Preference recovery:")
                print_tracking_block(relp_lag_family[m][fam][g], relp_thresholds, label="RelPE")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Bayesian preference-change detection models (train+test) with KL and Wasserstein scores."
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


def main() -> None:
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

    print("==============================================")
    print("PREFERENCE-CHANGE EVALUATION (KL + WASSERSTEIN)")
    print(f"Train dir        : {args.train_data_dir}")
    print(f"Test dir         : {args.test_data_dir}")
    print(f"Warmup steps     : {args.warmup_steps}")
    print(f"PD thresholds    : {pd_thresholds}")
    print(f"RelPE thresholds : {relp_thresholds}")
    print(f"RelPE mode       : {args.relp_mode} | eps={args.relp_eps}")
    print(f"MTL cap          : {args.cap_lag}")
    print(f"Models           : {', '.join(models_to_run)}")
    print("==============================================")

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
