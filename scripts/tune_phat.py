#!/usr/bin/env python3
"""
MPI + Optuna dual-objective tuning for self-directed users.

Workflow per model:
1) Tune on <self_root>/<MODEL>/train using Optuna multi-objective:
   - objective[0]: RelPE (l1_ratio), minimized
   - objective[1]: MSE, minimized
2) Select one trial from Pareto front:
   - choose the trial with the lowest objective[0] (RelPE l1_ratio),
   - break ties by objective[1] (MSE).
3) Evaluate chosen hyperparameters on <self_root>/<MODEL>/test and report:
   - paper metrics: MSE, ROC, PR, TR/Lag, MTL
   - diagnostics: MAE, PD, RelPE
   - overall, per-folder, theta-vs-p

MPI strategy:
- Data-parallel evaluation: each rank processes a shard of users.
- Rank 0 owns Optuna study (ask/tell) and broadcasts trial params.
- All ranks contribute metric aggregates with MPI reductions/gathers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
try:
    from mpi4py import MPI
except Exception as e:
    raise SystemExit(
        "Failed to import mpi4py/MPI runtime. "
        "Load an MPI module first (e.g., openmpi) and rerun.\n"
        f"Original error: {e}"
    )

from evaluate_phat import (
    ALL_MODELS,
    PAPER_TO_KEY,
    DEFAULT_CAP_LAG,
    DEFAULT_PD_THRESHOLDS,
    DEFAULT_RELP_EPS,
    DEFAULT_RELP_THRESHOLDS,
    DEFAULT_WARMUP_STEPS,
    MODEL_DISPLAY,
    capped_mtl_from_lags,
    compute_traces,
    folder_group,
    fmt,
    fmt_lag,
    fmt_pct,
    list_user_paths_with_folder,
    load_user_file_self_directed,
    nanmean_list,
    paper_threshold,
    tracking_lags_for_user,
    user_metrics,
)


OBJECTIVE_RELP_MODE = "l1_ratio"
MODEL_CHOICES = list(ALL_MODELS)


@dataclass
class UserRecord:
    path: str
    folder: str
    X: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    P: np.ndarray


def shard_list(items: List[Any], rank: int, size: int) -> List[Any]:
    return [items[i] for i in range(len(items)) if (i % size) == rank]


def parse_thresholds(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    parts = s.replace(",", " ").split()
    return [float(p) for p in parts]


def select_models(arg_models: List[str]) -> List[str]:
    if "all" in [m.lower() for m in arg_models]:
        return list(ALL_MODELS)

    selected: List[str] = []
    for m in arg_models:
        mk = PAPER_TO_KEY.get(m, m)  # accept paper names (KF-AF, BLR-VB, …)
        if mk in ALL_MODELS:
            selected.append(mk)
        else:
            print(f"[WARN] Unknown model '{m}' (skipping)")
    if not selected:
        raise SystemExit(f"No valid models selected. Options: {', '.join(ALL_MODELS)}")
    return selected


def suggest_params(trial: optuna.Trial, model_key: str) -> Dict[str, float]:
    if model_key == "KF":
        return {
            "variance_p": trial.suggest_float("variance_p", 1e-2, 5e1, log=True),
            "variance": trial.suggest_float("variance", 1e-2, 5e1, log=True),
            "delta": trial.suggest_float("delta", 1e-3, 0.3, log=True),
            "eta": trial.suggest_float("eta", 1e-7, 1e-1, log=True),
        }
    if model_key == "BLR":
        return {
            "default_variance": trial.suggest_float("default_variance", 1e-2, 1e2, log=True),
            "noise_precision": trial.suggest_float("noise_precision", 1e-2, 3e1, log=True),
        }
    if model_key == "vbBLR":
        return {
            "default_variance": trial.suggest_float("default_variance", 1e-2, 1e2, log=True),
            "noise_precision": trial.suggest_float("noise_precision", 1e-2, 3e1, log=True),
            "tau": trial.suggest_float("tau", 0.3, 1.0, log=True),
        }
    if model_key == "fBLR":
        return {
            "default_variance": trial.suggest_float("default_variance", 1e-2, 5e1, log=True),
            "noise_precision": trial.suggest_float("noise_precision", 1e-1, 3e1, log=True),
            "rho": trial.suggest_float("rho", 0.95, 0.999),
        }
    if model_key == "AROW":
        return {
            "lam1": trial.suggest_float("lam1", 1e-2, 1e2, log=True),
            "lam2": trial.suggest_float("lam2", 1e-2, 1e2, log=True),
        }
    if model_key == "BLRsw":
        return {
            "default_variance": trial.suggest_float("default_variance", 1e-2, 1e2, log=True),
            "noise_precision": trial.suggest_float("noise_precision", 1e-2, 3e1, log=True),
            "m": trial.suggest_int("m", 2, 50),
        }
    if model_key == "PBLR":
        return {
            "alpha": trial.suggest_float("alpha", 0.85, 1.0),
            "default_variance": trial.suggest_float("default_variance", 1e-2, 1e2, log=True),
            "noise_precision": trial.suggest_float("noise_precision", 1e-2, 3e1, log=True),
        }
    if model_key == "NIG":
        return {
            "default_variance": trial.suggest_float("default_variance", 1e-2, 1e2, log=True),
            "default_a": trial.suggest_float("default_a", 1.01, 50.0, log=True),
            "default_b": trial.suggest_float("default_b", 1e-3, 50.0, log=True),
        }
    raise ValueError(f"Unknown model key: {model_key}")


def load_users_sharded(
    data_dir: str,
    comm: MPI.Comm,
) -> Tuple[List[UserRecord], int]:
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        user_items = list_user_paths_with_folder(data_dir)
    else:
        user_items = None

    user_items = comm.bcast(user_items, root=0)
    my_items = shard_list(user_items, rank, size)

    users: List[UserRecord] = []
    for path, folder in my_items:
        try:
            X, y, labels, P = load_user_file_self_directed(path)
        except Exception:
            continue
        users.append(UserRecord(path=path, folder=folder, X=X, y=y, labels=labels, P=P))

    global_users = comm.allreduce(len(users), op=MPI.SUM)
    return users, int(global_users)


def local_objective_sums(
    users: List[UserRecord],
    model_key: str,
    params: Dict[str, float],
    warmup: int,
    relp_eps: float,
) -> Tuple[float, float, int]:
    relp_sum = 0.0
    mse_sum = 0.0
    used = 0

    for u in users:
        if u.X.shape[0] <= warmup:
            continue
        traces = compute_traces(
            model_key=model_key,
            X=u.X,
            y=u.y,
            params=params,
            P_true=u.P,
            relp_eps=relp_eps,
            relp_mode=OBJECTIVE_RELP_MODE,
        )
        met = user_metrics(traces, u.labels, warmup)
        if not met:
            continue

        relp_val = float(met.get("relp", float("nan")))
        mse_val = float(met.get("mse", float("nan")))
        if not (math.isfinite(relp_val) and math.isfinite(mse_val)):
            continue

        relp_sum += relp_val
        mse_sum += mse_val
        used += 1

    return float(relp_sum), float(mse_sum), int(used)


def run_multiobjective_optuna_mpi(
    comm: MPI.Comm,
    model_key: str,
    train_users: List[UserRecord],
    n_trials: int,
    seed: int,
    warmup: int,
    relp_eps: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    rank = comm.Get_rank()

    if rank == 0:
        sampler = optuna.samplers.NSGAIISampler(seed=int(seed))
        study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    else:
        study = None

    for _ in range(int(n_trials)):
        if rank == 0:
            trial = study.ask()
            params = suggest_params(trial, model_key)
        else:
            params = None

        params = comm.bcast(params, root=0)

        relp_local, mse_local, used_local = local_objective_sums(
            users=train_users,
            model_key=model_key,
            params=params,
            warmup=warmup,
            relp_eps=relp_eps,
        )

        relp_total = comm.reduce(relp_local, op=MPI.SUM, root=0)
        mse_total = comm.reduce(mse_local, op=MPI.SUM, root=0)
        used_total = comm.reduce(used_local, op=MPI.SUM, root=0)

        if rank == 0:
            if used_total <= 0:
                values = [1e18, 1e18]
            else:
                values = [float(relp_total / used_total), float(mse_total / used_total)]
            trial.set_user_attr("users_used", int(used_total))
            study.tell(trial, values)

    if rank == 0:
        completed = [t for t in study.trials if t.values is not None]
        if not completed:
            raise SystemExit(f"No completed trials for model {model_key}.")
        pareto = study.best_trials if study.best_trials else completed
        chosen = min(pareto, key=lambda t: (float(t.values[0]), float(t.values[1])))
        chosen_params = dict(chosen.params)
        summary = {
            "pareto_size": int(len(pareto)),
            "trial_number": int(chosen.number),
            "train_relp_l1_ratio": float(chosen.values[0]),
            "train_mse": float(chosen.values[1]),
            "chosen_params": chosen_params,
        }
    else:
        chosen_params = None
        summary = None

    chosen_params = comm.bcast(chosen_params, root=0)
    summary = comm.bcast(summary, root=0)
    return chosen_params, summary


def init_bucket() -> Dict[str, List[float]]:
    return {"roc_auc": [], "pr_auc": [], "mae": [], "mse": [], "pd": [], "relp": []}


def init_lag_block(thresholds: List[float]) -> Dict[float, Dict[str, Any]]:
    return {th: {"lags": [], "total": 0} for th in thresholds}


def local_eval_payload(
    users: List[UserRecord],
    model_key: str,
    params: Dict[str, float],
    warmup: int,
    pd_thresholds: List[float],
    relp_thresholds: List[float],
    relp_eps: float,
    relp_mode: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "used": 0,
        "skipped": 0,
        "overall": init_bucket(),
        "folder_metrics": {},
        "group_metrics": {"theta": init_bucket(), "p": init_bucket()},
        "pd_lag_overall": init_lag_block(pd_thresholds),
        "relp_lag_overall": init_lag_block(relp_thresholds),
        "pd_lag_folder": {},
        "relp_lag_folder": {},
        "pd_lag_group": {"theta": init_lag_block(pd_thresholds), "p": init_lag_block(pd_thresholds)},
        "relp_lag_group": {"theta": init_lag_block(relp_thresholds), "p": init_lag_block(relp_thresholds)},
    }

    for u in users:
        if u.X.shape[0] <= warmup:
            payload["skipped"] += 1
            continue

        traces = compute_traces(
            model_key=model_key,
            X=u.X,
            y=u.y,
            params=params,
            P_true=u.P,
            relp_eps=relp_eps,
            relp_mode=relp_mode,
        )
        met = user_metrics(traces, u.labels, warmup)
        if not met:
            payload["skipped"] += 1
            continue

        payload["used"] += 1
        for k in payload["overall"]:
            payload["overall"][k].append(float(met[k]))

        folder = u.folder
        if folder not in payload["folder_metrics"]:
            payload["folder_metrics"][folder] = init_bucket()
        for k in payload["folder_metrics"][folder]:
            payload["folder_metrics"][folder][k].append(float(met[k]))

        grp = folder_group(folder)
        if grp in ("theta", "p"):
            for k in payload["group_metrics"][grp]:
                payload["group_metrics"][grp][k].append(float(met[k]))

        user_pd_lags = tracking_lags_for_user(u.labels, traces["pd"], warmup, pd_thresholds)
        for th in pd_thresholds:
            lags, total = user_pd_lags[th]
            payload["pd_lag_overall"][th]["lags"].extend(lags)
            payload["pd_lag_overall"][th]["total"] += total

            if folder not in payload["pd_lag_folder"]:
                payload["pd_lag_folder"][folder] = init_lag_block(pd_thresholds)
            payload["pd_lag_folder"][folder][th]["lags"].extend(lags)
            payload["pd_lag_folder"][folder][th]["total"] += total

            if grp in ("theta", "p"):
                payload["pd_lag_group"][grp][th]["lags"].extend(lags)
                payload["pd_lag_group"][grp][th]["total"] += total

        user_relp_lags = tracking_lags_for_user(u.labels, traces["relp"], warmup, relp_thresholds)
        for th in relp_thresholds:
            lags, total = user_relp_lags[th]
            payload["relp_lag_overall"][th]["lags"].extend(lags)
            payload["relp_lag_overall"][th]["total"] += total

            if folder not in payload["relp_lag_folder"]:
                payload["relp_lag_folder"][folder] = init_lag_block(relp_thresholds)
            payload["relp_lag_folder"][folder][th]["lags"].extend(lags)
            payload["relp_lag_folder"][folder][th]["total"] += total

            if grp in ("theta", "p"):
                payload["relp_lag_group"][grp][th]["lags"].extend(lags)
                payload["relp_lag_group"][grp][th]["total"] += total

    return payload


def merge_bucket(dst: Dict[str, List[float]], src: Dict[str, List[float]]) -> None:
    for k in dst:
        dst[k].extend(src[k])


def merge_lag_block(dst: Dict[float, Dict[str, Any]], src: Dict[float, Dict[str, Any]], thresholds: List[float]) -> None:
    for th in thresholds:
        dst[th]["lags"].extend(src[th]["lags"])
        dst[th]["total"] += int(src[th]["total"])


def merge_eval_payloads(payloads: List[Dict[str, Any]], pd_thresholds: List[float], relp_thresholds: List[float]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "used": 0,
        "skipped": 0,
        "overall": init_bucket(),
        "folder_metrics": {},
        "group_metrics": {"theta": init_bucket(), "p": init_bucket()},
        "pd_lag_overall": init_lag_block(pd_thresholds),
        "relp_lag_overall": init_lag_block(relp_thresholds),
        "pd_lag_folder": {},
        "relp_lag_folder": {},
        "pd_lag_group": {"theta": init_lag_block(pd_thresholds), "p": init_lag_block(pd_thresholds)},
        "relp_lag_group": {"theta": init_lag_block(relp_thresholds), "p": init_lag_block(relp_thresholds)},
    }

    for pl in payloads:
        merged["used"] += int(pl["used"])
        merged["skipped"] += int(pl["skipped"])
        merge_bucket(merged["overall"], pl["overall"])

        for folder, bucket in pl["folder_metrics"].items():
            if folder not in merged["folder_metrics"]:
                merged["folder_metrics"][folder] = init_bucket()
            merge_bucket(merged["folder_metrics"][folder], bucket)

        for grp in ("theta", "p"):
            merge_bucket(merged["group_metrics"][grp], pl["group_metrics"][grp])

        merge_lag_block(merged["pd_lag_overall"], pl["pd_lag_overall"], pd_thresholds)
        merge_lag_block(merged["relp_lag_overall"], pl["relp_lag_overall"], relp_thresholds)

        for folder, lagblk in pl["pd_lag_folder"].items():
            if folder not in merged["pd_lag_folder"]:
                merged["pd_lag_folder"][folder] = init_lag_block(pd_thresholds)
            merge_lag_block(merged["pd_lag_folder"][folder], lagblk, pd_thresholds)

        for folder, lagblk in pl["relp_lag_folder"].items():
            if folder not in merged["relp_lag_folder"]:
                merged["relp_lag_folder"][folder] = init_lag_block(relp_thresholds)
            merge_lag_block(merged["relp_lag_folder"][folder], lagblk, relp_thresholds)

        for grp in ("theta", "p"):
            merge_lag_block(merged["pd_lag_group"][grp], pl["pd_lag_group"][grp], pd_thresholds)
            merge_lag_block(merged["relp_lag_group"][grp], pl["relp_lag_group"][grp], relp_thresholds)

    return merged


def lag_report(lag_dict: Dict[float, Dict[str, Any]], thresholds: List[float], cap_lag: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for th in thresholds:
        lags = lag_dict[th]["lags"]
        total = int(lag_dict[th]["total"])
        rec = len(lags)
        tr = (rec / total) if total else float("nan")
        lag = float(np.mean(lags)) if rec else (float("inf") if total else float("nan"))
        mtl = capped_mtl_from_lags(lags, total, cap_lag)
        out[f"{th:g}"] = {
            "tr": float(tr),
            "lag": float(lag),
            "mtl": float(mtl),
            "recovered_events": int(rec),
            "total_events": int(total),
        }
    return out


def bucket_means(bucket: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        "roc_auc": nanmean_list(bucket["roc_auc"]),
        "pr_auc": nanmean_list(bucket["pr_auc"]),
        "mae": nanmean_list(bucket["mae"]),
        "mse": nanmean_list(bucket["mse"]),
        "pd": nanmean_list(bucket["pd"]),
        "relp": nanmean_list(bucket["relp"]),
    }


def build_eval_report(
    merged: Dict[str, Any],
    pd_thresholds: List[float],
    relp_thresholds: List[float],
    cap_lag: int,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "users_used": int(merged["used"]),
        "users_skipped": int(merged["skipped"]),
        "overall": bucket_means(merged["overall"]),
        "overall_tracking_pd": lag_report(merged["pd_lag_overall"], pd_thresholds, cap_lag),
        "overall_tracking_relp": lag_report(merged["relp_lag_overall"], relp_thresholds, cap_lag),
        "per_folder": {},
        "groups": {},
    }

    for folder in sorted(merged["folder_metrics"].keys()):
        report["per_folder"][folder] = {
            "metrics": bucket_means(merged["folder_metrics"][folder]),
            "tracking_pd": lag_report(merged["pd_lag_folder"].get(folder, init_lag_block(pd_thresholds)), pd_thresholds, cap_lag),
            "tracking_relp": lag_report(
                merged["relp_lag_folder"].get(folder, init_lag_block(relp_thresholds)),
                relp_thresholds,
                cap_lag,
            ),
        }

    for grp in ("theta", "p"):
        report["groups"][grp] = {
            "metrics": bucket_means(merged["group_metrics"][grp]),
            "users": int(len(merged["group_metrics"][grp]["roc_auc"])),
            "tracking_pd": lag_report(merged["pd_lag_group"][grp], pd_thresholds, cap_lag),
            "tracking_relp": lag_report(merged["relp_lag_group"][grp], relp_thresholds, cap_lag),
        }

    return report


def print_tracking_from_report(tr: Dict[str, Dict[str, float]], thresholds: List[float]) -> None:
    for th in thresholds:
        row = tr[f"{th:g}"]
        print(
            f"    RelPE<={th:0.2f}: "
            f"TR={fmt_pct(row['tr'])} | "
            f"Lag={fmt_lag(row['lag'])} | "
            f"MTL={fmt_lag(row['mtl'])} | "
            f"Events={int(row['recovered_events'])}/{int(row['total_events'])}"
        )


def paper_report_line(metrics: Dict[str, float], tracking: Dict[str, Dict[str, float]], threshold: float) -> str:
    row = tracking[f"{threshold:g}"]
    return (
        f"MSE={fmt(metrics['mse'], 3)} | "
        f"ROC={fmt_pct(metrics['roc_auc'])} | "
        f"PR={fmt_pct(metrics['pr_auc'])} | "
        f"TR/Lag={fmt_pct(row['tr'])}/{fmt_lag(row['lag'])} | "
        f"MTL={fmt_lag(row['mtl'])}"
    )


def print_eval_report(
    model_key: str,
    report: Dict[str, Any],
    warmup: int,
    pd_thresholds: List[float],
    relp_thresholds: List[float],
    relp_mode: str,
) -> None:
    ov = report["overall"]
    paper_recovery_threshold = paper_threshold(relp_thresholds)
    print(f"=== TEST Evaluation (macro over users; from timestep {warmup + 1}) ===")
    print(f"{MODEL_DISPLAY[model_key]} ({model_key}):")
    print(f"  Users used: {report['users_used']} | skipped: {report['users_skipped']}")
    print(f"  Paper metrics: {paper_report_line(ov, report['overall_tracking_relp'], paper_recovery_threshold)}")
    print(f"  Diagnostics: MAE={fmt(ov['mae'], 6)} | PD={fmt(ov['pd'], 6)} | RelPE={fmt(ov['relp'], 6)}")
    print(f"  Preference recovery: RelPE={relp_mode}")
    print_tracking_from_report(report["overall_tracking_relp"], relp_thresholds)
    print()

    print("=== Per-folder (macro over users in folder) ===")
    if not report["per_folder"]:
        print("  (no evaluable users)")
    else:
        for folder in sorted(report["per_folder"].keys()):
            m = report["per_folder"][folder]["metrics"]
            print(f"  - {folder}: {paper_report_line(m, report['per_folder'][folder]['tracking_relp'], paper_recovery_threshold)}")
            print(f"    Diagnostics: MAE={fmt(m['mae'], 6)} | PD={fmt(m['pd'], 6)} | RelPE={fmt(m['relp'], 6)}")
            print("    Preference recovery:")
            print_tracking_from_report(report["per_folder"][folder]["tracking_relp"], relp_thresholds)
    print()

    print("=== THETA vs P (macro over users) ===")
    for grp in ("theta", "p"):
        g = report["groups"][grp]
        m = g["metrics"]
        print(f"  {grp.upper():9s}: {paper_report_line(m, g['tracking_relp'], paper_recovery_threshold)} (users: {g['users']})")
        print(f"    Diagnostics: MAE={fmt(m['mae'], 6)} | PD={fmt(m['pd'], 6)} | RelPE={fmt(m['relp'], 6)}")
        print("    Preference recovery:")
        print_tracking_from_report(g["tracking_relp"], relp_thresholds)
    print()


def evaluate_on_test_mpi(
    comm: MPI.Comm,
    test_users: List[UserRecord],
    model_key: str,
    params: Dict[str, float],
    warmup: int,
    pd_thresholds: List[float],
    relp_thresholds: List[float],
    relp_eps: float,
    relp_mode: str,
    cap_lag: int,
) -> Optional[Dict[str, Any]]:
    rank = comm.Get_rank()
    local = local_eval_payload(
        users=test_users,
        model_key=model_key,
        params=params,
        warmup=warmup,
        pd_thresholds=pd_thresholds,
        relp_thresholds=relp_thresholds,
        relp_eps=relp_eps,
        relp_mode=relp_mode,
    )
    gathered = comm.gather(local, root=0)

    if rank != 0:
        return None

    merged = merge_eval_payloads(gathered, pd_thresholds, relp_thresholds)
    return build_eval_report(merged, pd_thresholds, relp_thresholds, cap_lag)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MPI Optuna dual-objective tuning on self-directed train users, then evaluate on test users."
    )
    p.add_argument("--self-root", required=True, help="Root directory containing per-model folders with train/test.")
    p.add_argument("--models", "-m", nargs="+", default=["all"], help="Models to run. Use 'all' or keys (KF BLR ...).")
    p.add_argument("--list-models", "-l", action="store_true", help="List available models and exit.")
    p.add_argument("--n-trials", type=int, default=200, help="Optuna trials per model.")
    p.add_argument("--seed", type=int, default=3, help="Random seed for Optuna sampler.")
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Warmup steps for objective and eval.")

    p.add_argument("--pd-thresholds", type=str, default="", help="PD thresholds, e.g. '0.05,0.1,0.2,...'")
    p.add_argument("--relp-thresholds", type=str, default="", help="RelPE thresholds for tracking lag. The paper uses 0.25.")
    p.add_argument("--relp-eps", type=float, default=DEFAULT_RELP_EPS, help="Epsilon for RelPE denominator.")
    p.add_argument(
        "--eval-relp-mode",
        type=str,
        default=OBJECTIVE_RELP_MODE,
        choices=["mean_coord", "l1_ratio", "l2_ratio"],
        help="RelPE scalarization mode for final TEST evaluation outputs.",
    )
    p.add_argument("--cap-lag", type=int, default=DEFAULT_CAP_LAG, help="Penalty assigned to non-recovered events when computing paper MTL.")
    p.add_argument("--quiet-optuna", action="store_true", help="Reduce Optuna logging verbosity.")
    p.add_argument("--output-json", type=str, default="", help="Optional output JSON path for tuned params + test metrics.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.quiet_optuna:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.list_models:
        if rank == 0:
            print("Available models:")
            for m in ALL_MODELS:
                print(f"  {m:6s} - {MODEL_DISPLAY[m]}")
        return

    models_to_run = select_models(args.models)
    pd_thresholds = parse_thresholds(args.pd_thresholds) if args.pd_thresholds.strip() else DEFAULT_PD_THRESHOLDS
    relp_thresholds = parse_thresholds(args.relp_thresholds) if args.relp_thresholds.strip() else DEFAULT_RELP_THRESHOLDS

    if args.n_trials <= 0:
        raise SystemExit("--n-trials must be > 0.")

    if rank == 0:
        print("======================================================")
        print("SELF-DIRECTED MPI OPTUNA (DUAL OBJECTIVE)")
        print(f"MPI ranks          : {size}")
        print(f"Self root          : {args.self_root}")
        print(f"Models             : {', '.join(models_to_run)}")
        print(f"Trials/model       : {args.n_trials}")
        print(f"Warmup steps       : {args.warmup_steps}")
        print(f"Objective 1        : RelPE ({OBJECTIVE_RELP_MODE})")
        print("Objective 2        : MSE")
        print(f"Pareto selection   : min RelPE ({OBJECTIVE_RELP_MODE}), tie-break min MSE")
        print(f"TEST eval RelPE mode: {args.eval_relp_mode} | eps={args.relp_eps}")
        print(f"PD thresholds      : {pd_thresholds}")
        print(f"RelPE thresholds   : {relp_thresholds}")
        print(f"MTL cap            : {args.cap_lag}")
        print("======================================================")
        print()

    all_results: Dict[str, Any] = {}

    for model_key in models_to_run:
        train_dir = os.path.join(args.self_root, model_key, "train")
        test_dir = os.path.join(args.self_root, model_key, "test")

        if rank == 0:
            can_run = os.path.isdir(train_dir) and os.path.isdir(test_dir)
            if not can_run:
                print(f"[WARN] Missing train/test for model {model_key}:")
                print(f"       train={train_dir}")
                print(f"       test ={test_dir}")
        else:
            can_run = None
        can_run = comm.bcast(can_run, root=0)
        if not can_run:
            continue

        train_users, n_train_global = load_users_sharded(train_dir, comm)
        if rank == 0:
            print("------------------------------------------------------")
            print(f"MODEL {model_key}: {MODEL_DISPLAY[model_key]}")
            print(f"Train users loaded (global, valid): {n_train_global}")
            print(f"Train directory: {train_dir}")

        if n_train_global <= 0:
            if rank == 0:
                print(f"[WARN] No valid train users for model {model_key}. Skipping.\n")
            continue

        tuned_params, tuning_summary = run_multiobjective_optuna_mpi(
            comm=comm,
            model_key=model_key,
            train_users=train_users,
            n_trials=int(args.n_trials),
            seed=int(args.seed),
            warmup=int(args.warmup_steps),
            relp_eps=float(args.relp_eps),
        )

        test_users, n_test_global = load_users_sharded(test_dir, comm)
        if rank == 0:
            print(f"Test users loaded (global, valid):  {n_test_global}")
            print(f"Test directory: {test_dir}")

        eval_report = evaluate_on_test_mpi(
            comm=comm,
            test_users=test_users,
            model_key=model_key,
            params=tuned_params,
            warmup=int(args.warmup_steps),
            pd_thresholds=pd_thresholds,
            relp_thresholds=relp_thresholds,
            relp_eps=float(args.relp_eps),
            relp_mode=args.eval_relp_mode,
            cap_lag=int(args.cap_lag),
        )

        if rank == 0:
            print("=== Tuned Hyperparameters (selected from Pareto) ===")
            print(
                f"  train RelPE(l1_ratio)={fmt(tuning_summary['train_relp_l1_ratio'], 6)} | "
                f"train MSE={fmt(tuning_summary['train_mse'], 6)} | "
                f"pareto_size={tuning_summary['pareto_size']} | "
                f"trial={tuning_summary['trial_number']}"
            )
            for k, v in tuned_params.items():
                print(f"  {k}: {v}")
            print()

            if eval_report is None:
                print("[WARN] No evaluation report generated.\n")
                continue

            print_eval_report(
                model_key=model_key,
                report=eval_report,
                warmup=int(args.warmup_steps),
                pd_thresholds=pd_thresholds,
                relp_thresholds=relp_thresholds,
                relp_mode=args.eval_relp_mode,
            )

            all_results[model_key] = {
                "model_display": MODEL_DISPLAY[model_key],
                "train_dir": train_dir,
                "test_dir": test_dir,
                "objective": {
                    "objective_1": f"RelPE ({OBJECTIVE_RELP_MODE})",
                    "objective_2": "MSE",
                    "selection_rule": "min objective_1, tie-break min objective_2",
                },
                "tuning": tuning_summary,
                "test_eval": eval_report,
            }

    if rank == 0 and args.output_json.strip():
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved report JSON: {out_path}")


if __name__ == "__main__":
    main()
