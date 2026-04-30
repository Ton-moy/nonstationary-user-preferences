#!/usr/bin/env python3
"""
Appendix Tables 2 and 3 — full synthetic benchmark evaluation.

Table 2: One section per setting (θ-driven, p-driven, p̂-driven top-2%).
         Metrics aggregated over all 6 scenarios within each setting.
Table 3: p̂-driven mixed selection, one section per scenario: PS, PSC1T, PB, PBC1T.

Hyperparameters: configs/model_hyperparameters_synthetic.json (group: evaluation)
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_phat import (
    DEFAULT_CAP_LAG,
    DEFAULT_PARAMS_BY_MODEL,
    DEFAULT_RELP_EPS,
    DEFAULT_WARMUP_STEPS,
    MODEL_DISPLAY,
    compute_traces,
    fmt,
    fmt_lag,
    fmt_pct,
    load_params_by_model,
    load_user_file_self_directed,
    nanmean_list,
    paper_tracking_values,
    tracking_lags_for_user,
    user_metrics,
)

# Paper-canonical model order
PAPER_MODEL_ORDER = ["KF", "AROW", "BLR", "vbBLR", "fBLR", "BLRsw", "PBLR", "NIG"]

# Maps model key → data directory name for p̂-driven datasets
MODEL_DATA_DIR = {
    "KF":    "KF-AF",
    "AROW":  "AROW",
    "BLR":   "BLR",
    "vbBLR": "BLR-VB",
    "fBLR":  "BLR-FF",
    "BLRsw": "BLR-SW",
    "PBLR":  "BLR-PP",
    "NIG":   "BLR-NIG",
}

# All 6 scenario subdirectories under theta_driven/test/ and p_driven/test/
ALL_SCENARIOS = ["ps", "psc1t", "psc2t", "pb", "pbc1t", "pbc2t"]

# Appendix Table 3: 4 scenarios shown.
# PSC and PBC rows combine all conservation sub-folders (1-topic + 2-topic),
# matching the PS_PLUS_PC / PB_PLUS_PC aggregates from the original evaluation.
TABLE3_SCENARIOS: Dict[str, Tuple[str, ...]] = {
    "PS":    ("ps_theta",    "ps_p"),
    "PSC1T": ("psc1t_theta", "psc1t_p", "psc2t_theta", "psc2t_p"),
    "PB":    ("pb_theta",    "pb_p"),
    "PBC1T": ("pbc1t_theta", "pbc1t_p", "pbc2t_theta", "pbc2t_p"),
}

DEFAULT_RELPE_MODE = "l1_ratio"

TABLE_HEADERS = ["Model", "MSE", "ROC %", "PR %", "TR %", "Lag", "MTL"]
COL_WIDTHS    = [8,       7,     7,       7,       7,      5,     5]
SECTION_DIVIDER = "=" * 72


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_theta_p_files(data_dir: Path) -> List[Path]:
    """All JSON files across all 6 scenario subdirs under theta/p-driven test root."""
    files: List[Path] = []
    for scenario in ALL_SCENARIOS:
        scenario_dir = data_dir / scenario
        if scenario_dir.is_dir():
            files.extend(sorted(scenario_dir.glob("*.json")))
    return files


def _model_base(phat_root: Path, model_key: str) -> Path:
    """Resolve the base directory for a model inside a p̂-driven root."""
    model_dir = phat_root / MODEL_DATA_DIR[model_key]
    for candidate in [model_dir / "all", model_dir]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No directory found for {MODEL_DISPLAY[model_key]} under {phat_root}"
    )


def collect_phat_all_files(phat_root: Path, model_key: str) -> List[Path]:
    """All JSON files across all scenario sub-folders for one model (Table 2 aggregate)."""
    base = _model_base(phat_root, model_key)
    files: List[Path] = []
    for sub in sorted(base.iterdir()):
        if sub.is_dir():
            files.extend(sorted(sub.glob("*.json")))
    return files


def collect_phat_scenario_files(
    phat_root: Path, model_key: str, sub_folders: Tuple[str, ...]
) -> List[Path]:
    """JSON files for a specific scenario (theta + p sub-folders) for one model (Table 3)."""
    base = _model_base(phat_root, model_key)
    files: List[Path] = []
    for folder in sub_folders:
        folder_path = base / folder
        if folder_path.is_dir():
            files.extend(sorted(folder_path.glob("*.json")))
    return files


def collect_phat_group_files(phat_root: Path, model_key: str, suffix: str) -> List[Path]:
    """JSON files from all sub-folders whose name ends with `suffix` (e.g. '_theta' or '_p')."""
    base = _model_base(phat_root, model_key)
    files: List[Path] = []
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and sub.name.endswith(suffix):
            files.extend(sorted(sub.glob("*.json")))
    return files


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_files(
    model_key: str,
    files: List[Path],
    params: Dict[str, float],
    warmup_steps: int,
    relpe_threshold: float,
    relpe_eps: float,
    relpe_mode: str,
    cap_lag: int,
    x_key: str = "generated_topic_vector",
) -> Dict[str, Any]:
    """Run model on a list of user files and aggregate metrics."""
    if not files:
        return _empty_result()

    bucket: Dict[str, List[float]] = {
        k: [] for k in ("roc_auc", "pr_auc", "mae", "mse", "pd", "relp")
    }
    tracking: Dict[str, Any] = {"lags": [], "total": 0}
    users = 0

    for path in files:
        X, y, labels, P = load_user_file_self_directed(str(path), x_key=x_key)
        if X.shape[0] <= warmup_steps:
            continue

        traces = compute_traces(
            model_key, X, y, params, P, relp_eps=relpe_eps, relp_mode=relpe_mode,
        )
        metrics = user_metrics(traces, labels, warmup_steps)
        if not metrics:
            continue

        users += 1
        for key in bucket:
            bucket[key].append(metrics[key])

        user_lags = tracking_lags_for_user(
            labels, traces["relp"], warmup_steps, [relpe_threshold],
            max_lag_horizon=cap_lag,
        )
        lags, total = user_lags[relpe_threshold]
        tracking["lags"].extend(lags)
        tracking["total"] += total

    result = _empty_result()
    result["metrics"] = {k: nanmean_list(v) for k, v in bucket.items()}
    result["tracking"] = tracking
    result["users"] = users
    return result


def _empty_result() -> Dict[str, Any]:
    return {
        "metrics": {k: float("nan") for k in ("roc_auc", "pr_auc", "mae", "mse", "pd", "relp")},
        "tracking": {"lags": [], "total": 0},
        "users": 0,
    }


def format_result(
    result: Dict[str, Any], relpe_threshold: float, cap_lag: int
) -> Dict[str, str]:
    lag_block = {relpe_threshold: result["tracking"]}
    tr, lag, mtl, _, _ = paper_tracking_values(lag_block, relpe_threshold, cap_lag)
    m = result["metrics"]
    return {
        "MSE":   fmt(m["mse"], 4),
        "ROC %": fmt_pct(m["roc_auc"]),
        "PR %":  fmt_pct(m["pr_auc"]),
        "TR %":  fmt_pct(tr),
        "Lag":   fmt_lag(lag),
        "MTL":   fmt_lag(mtl),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def section_table(rows: List[Dict[str, str]], title: str) -> str:
    header_cells = [v.rjust(COL_WIDTHS[i]) for i, v in enumerate(TABLE_HEADERS)]
    sep = "  ".join("-" * w for w in COL_WIDTHS)
    lines = [
        SECTION_DIVIDER,
        f"  {title}",
        SECTION_DIVIDER,
        "  ".join(header_cells),
        sep,
    ]
    for row in rows:
        cells = [row[h].rjust(COL_WIDTHS[i]) for i, h in enumerate(TABLE_HEADERS)]
        lines.append("  ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print Appendix Tables 2 and 3 of the synthetic benchmark."
    )
    p.add_argument("--theta-dir", type=Path,
                   default=REPO_ROOT / "data/synthetic/theta_driven/test",
                   help="Root test directory for θ-driven setting.")
    p.add_argument("--p-dir", type=Path,
                   default=REPO_ROOT / "data/synthetic/p_driven/test",
                   help="Root test directory for p-driven setting.")
    p.add_argument("--phat-top2-dir", type=Path,
                   default=REPO_ROOT / "data/synthetic/phat_driven_top2",
                   help="Root for p̂-driven top-2% data (one sub-folder per model).")
    p.add_argument("--phat-mixed-dir", type=Path,
                   default=REPO_ROOT / "data/synthetic/phat_driven_mixed",
                   help="Root for p̂-driven mixed data (one sub-folder per model).")
    p.add_argument("--warmup-steps",    type=int,   default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--relpe-threshold", type=float, default=0.25)
    p.add_argument("--relpe-eps",       type=float, default=DEFAULT_RELP_EPS)
    p.add_argument("--relpe-mode",      default=DEFAULT_RELPE_MODE,
                   choices=["mean_coord", "l1_ratio", "l2_ratio"])
    p.add_argument("--cap-lag",         type=int,   default=DEFAULT_CAP_LAG)
    p.add_argument("--params-by-model-json", default=None,
                   help="JSON string or path. Default: configs/model_hyperparameters_synthetic.json.")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/appendix")
    p.add_argument("--no-output-files", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    params_by_model = (
        load_params_by_model(args.params_by_model_json)
        if args.params_by_model_json
        else DEFAULT_PARAMS_BY_MODEL
    )

    ev = dict(
        warmup_steps=args.warmup_steps,
        relpe_threshold=args.relpe_threshold,
        relpe_eps=args.relpe_eps,
        relpe_mode=args.relpe_mode,
        cap_lag=args.cap_lag,
    )

    # ------------------------------------------------------------------
    # Table 2: three settings, each aggregated over all 6 scenarios
    # ------------------------------------------------------------------
    theta_rows: List[Dict[str, str]] = []
    p_rows:     List[Dict[str, str]] = []
    phat_rows:  List[Dict[str, str]] = []

    for model_key in PAPER_MODEL_ORDER:
        params  = params_by_model[model_key]
        display = MODEL_DISPLAY[model_key]

        theta_files = collect_theta_p_files(args.theta_dir)
        r = evaluate_files(model_key, theta_files, params, x_key="topic_vector", **ev)
        theta_rows.append({"Model": display, **format_result(r, args.relpe_threshold, args.cap_lag)})

        p_files = collect_theta_p_files(args.p_dir)
        r = evaluate_files(model_key, p_files, params, x_key="topic_vector", **ev)
        p_rows.append({"Model": display, **format_result(r, args.relpe_threshold, args.cap_lag)})

        phat_files = collect_phat_all_files(args.phat_top2_dir, model_key)
        r = evaluate_files(model_key, phat_files, params, x_key="generated_topic_vector", **ev)
        phat_rows.append({"Model": display, **format_result(r, args.relpe_threshold, args.cap_lag)})

    # ------------------------------------------------------------------
    # Table 3: p̂-driven mixed, 4 scenarios
    # ------------------------------------------------------------------
    t3_rows: Dict[str, List[Dict[str, str]]] = {s: [] for s in TABLE3_SCENARIOS}

    for model_key in PAPER_MODEL_ORDER:
        params  = params_by_model[model_key]
        display = MODEL_DISPLAY[model_key]

        for scenario_name, sub_folders in TABLE3_SCENARIOS.items():
            files = collect_phat_scenario_files(args.phat_mixed_dir, model_key, sub_folders)
            r = evaluate_files(model_key, files, params, x_key="generated_topic_vector", **ev)
            t3_rows[scenario_name].append(
                {"Model": display, **format_result(r, args.relpe_threshold, args.cap_lag)}
            )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    hp_note = "custom" if args.params_by_model_json else "configs/model_hyperparameters_synthetic.json"
    header = "\n".join([
        "Appendix Tables 2 and 3 — full synthetic benchmark",
        f"RelPE:  threshold={args.relpe_threshold:g}  mode={args.relpe_mode}",
        f"HPs:    {hp_note}",
        "",
    ])

    table2 = "\n\n".join([
        "Table 2  (all 6 scenarios aggregated per setting)",
        section_table(theta_rows, "θ-driven"),
        section_table(p_rows,     "p-driven"),
        section_table(phat_rows,  "p̂-driven  (top-2% selection)"),
    ]) + "\n" + SECTION_DIVIDER

    table3 = "\n\n".join([
        "\nTable 3  (p̂-driven mixed selection, per scenario)",
        section_table(t3_rows["PS"],    "PS"),
        section_table(t3_rows["PSC1T"], "PSC1T"),
        section_table(t3_rows["PB"],    "PB"),
        section_table(t3_rows["PBC1T"], "PBC1T"),
    ]) + "\n" + SECTION_DIVIDER

    output = header + table2 + "\n" + table3
    print(output)

    if not args.no_output_files:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        txt_path = args.output_dir / f"appendix_tables2_3_{stamp}.txt"
        csv_path = args.output_dir / f"appendix_tables2_3_{stamp}.csv"

        txt_path.write_text(output + "\n", encoding="utf-8")

        fieldnames = ["Table", "Section"] + TABLE_HEADERS
        # Flat CSV
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for label, section, rows in [
                ("Table2", "theta-driven",    theta_rows),
                ("Table2", "p-driven",        p_rows),
                ("Table2", "phat-top2",       phat_rows),
                ("Table3", "PS",              t3_rows["PS"]),
                ("Table3", "PSC1T",           t3_rows["PSC1T"]),
                ("Table3", "PB",              t3_rows["PB"]),
                ("Table3", "PBC1T",           t3_rows["PBC1T"]),
            ]:
                for row in rows:
                    writer.writerow({"Table": label, "Section": section, **row})

        print(f"\nWrote text : {txt_path}")
        print(f"Wrote CSV  : {csv_path}")


if __name__ == "__main__":
    main()
