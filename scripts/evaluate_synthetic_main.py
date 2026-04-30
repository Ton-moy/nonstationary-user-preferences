#!/usr/bin/env python3
"""
Main-paper synthetic evaluation table.

Prints Table 3-style results for the p-hat-driven mixed-selection setting.
Results are shown in two separate sections:
  - Preference Shifting (PS): ps_theta + ps_p
  - Preference Broadening (PB): pb_theta + pb_p

Models are displayed in paper order: KF-AF, AROW, BLR, BLR-VB, BLR-FF, BLR-SW, BLR-PP, BLR-NIG.
Columns: MSE | ROC | PR | TR | Lag | MTL
Hyperparameters: configs/model_hyperparameters_synthetic.json (group: evaluation)
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_phat import (
    ALL_MODELS,
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

# Paper-canonical model order (Table 3 / Appendix Table 2)
PAPER_MODEL_ORDER = ["KF", "AROW", "BLR", "vbBLR", "fBLR", "BLRsw", "PBLR", "NIG"]

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

SCENARIO_FOLDERS = {
    "PS": ("ps_theta", "ps_p"),
    "PB": ("pb_theta", "pb_p"),
}

DEFAULT_MAIN_RELPE_MODE = "l1_ratio"

# Section headers and separators
SECTION_DIVIDER = "=" * 72
SECTION_PS = "Preference Shifting (PS)"
SECTION_PB = "Preference Broadening (PB)"

TABLE_HEADERS = ["Model", "MSE", "ROC %", "PR %", "TR %", "Lag", "MTL"]
COL_WIDTHS    = [8,       7,     7,       7,       7,      5,     5]


def select_models(raw_models: Iterable[str]) -> List[str]:
    requested = list(raw_models)
    if any(m.lower() == "all" for m in requested):
        return list(PAPER_MODEL_ORDER)

    selected: List[str] = []
    display_to_key = {display: key for key, display in MODEL_DISPLAY.items()}
    data_dir_to_key = {dirname: key for key, dirname in MODEL_DATA_DIR.items()}

    for model in requested:
        if model in ALL_MODELS:
            selected.append(model)
        elif model in display_to_key:
            selected.append(display_to_key[model])
        elif model in data_dir_to_key:
            selected.append(data_dir_to_key[model])
        else:
            valid_display = ", ".join(MODEL_DATA_DIR.values())
            raise SystemExit(
                f"Unknown model '{model}'. Valid paper names: {valid_display}."
            )
    # Sort by paper order, keeping unknowns at the end
    order_map = {k: i for i, k in enumerate(PAPER_MODEL_ORDER)}
    selected.sort(key=lambda k: order_map.get(k, len(PAPER_MODEL_ORDER)))
    return selected


def model_root(data_dir: Path, model_key: str) -> Path:
    candidates = [
        data_dir / MODEL_DATA_DIR[model_key] / "all",
        data_dir / MODEL_DATA_DIR[model_key],
        data_dir / model_key / "all",
        data_dir / model_key,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find p-hat data for {MODEL_DISPLAY[model_key]} under {data_dir}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def scenario_files(root: Path, scenario: str) -> List[Path]:
    paths: List[Path] = []
    for folder in SCENARIO_FOLDERS[scenario]:
        folder_path = root / folder
        if folder_path.is_dir():
            paths.extend(sorted(folder_path.glob("*.json")))
    return paths


def init_metric_bucket() -> Dict[str, List[float]]:
    return {"roc_auc": [], "pr_auc": [], "mae": [], "mse": [], "pd": [], "relp": []}


def summarize_bucket(bucket: Dict[str, List[float]]) -> Dict[str, float]:
    return {key: nanmean_list(values) for key, values in bucket.items()}


def empty_result() -> Dict[str, Any]:
    return {
        "metrics": {key: float("nan") for key in ("roc_auc", "pr_auc", "mae", "mse", "pd", "relp")},
        "tracking": {"lags": [], "total": 0},
        "users": 0,
        "files": 0,
    }


def evaluate_scenario(
    model_key: str,
    files: List[Path],
    params: Dict[str, float],
    warmup_steps: int,
    relpe_threshold: float,
    relpe_eps: float,
    relpe_mode: str,
    cap_lag: int,
) -> Dict[str, Any]:
    if not files:
        return empty_result()

    bucket = init_metric_bucket()
    tracking = {relpe_threshold: {"lags": [], "total": 0}}
    users = 0

    for path in files:
        X, y, labels, P = load_user_file_self_directed(str(path))
        if X.shape[0] <= warmup_steps:
            continue

        traces = compute_traces(
            model_key,
            X,
            y,
            params,
            P,
            relp_eps=relpe_eps,
            relp_mode=relpe_mode,
        )
        metrics = user_metrics(traces, labels, warmup_steps)
        if not metrics:
            continue

        users += 1
        for key in bucket:
            bucket[key].append(metrics[key])

        user_lags = tracking_lags_for_user(
            labels,
            traces["relp"],
            warmup_steps,
            [relpe_threshold],
            max_lag_horizon=cap_lag,
        )
        lags, total = user_lags[relpe_threshold]
        tracking[relpe_threshold]["lags"].extend(lags)
        tracking[relpe_threshold]["total"] += total

    result = empty_result()
    result["metrics"] = summarize_bucket(bucket)
    result["tracking"] = tracking[relpe_threshold]
    result["users"] = users
    result["files"] = len(files)
    return result


def format_result(result: Dict[str, Any], relpe_threshold: float, cap_lag: int) -> Dict[str, str]:
    lag_block = {relpe_threshold: result["tracking"]}
    tr, lag, mtl, _, _ = paper_tracking_values(lag_block, relpe_threshold, cap_lag)
    m = result["metrics"]
    return {
        "MSE":   fmt(m["mse"], 3),
        "ROC %": fmt_pct(m["roc_auc"]),
        "PR %":  fmt_pct(m["pr_auc"]),
        "TR %":  fmt_pct(tr),
        "Lag":   fmt_lag(lag),
        "MTL":   fmt_lag(mtl),
    }


def _cell(value: str, width: int) -> str:
    return value.rjust(width)


def section_table(section_rows: List[Dict[str, str]], title: str) -> str:
    """Render one scenario section as a plain-text aligned table."""
    header_cells = [_cell(h, COL_WIDTHS[i]) for i, h in enumerate(TABLE_HEADERS)]
    sep = "  ".join("-" * w for w in COL_WIDTHS)
    header_line = "  ".join(header_cells)

    lines = [
        SECTION_DIVIDER,
        f"  {title}",
        SECTION_DIVIDER,
        header_line,
        sep,
    ]
    for row in section_rows:
        cells = [_cell(row[h], COL_WIDTHS[i]) for i, h in enumerate(TABLE_HEADERS)]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def markdown_two_sections(ps_rows: List[Dict[str, str]], pb_rows: List[Dict[str, str]]) -> str:
    """Render two separate markdown tables (PS then PB) for CSV/text output."""
    def md_table(rows: List[Dict[str, str]], title: str) -> str:
        headers = TABLE_HEADERS
        divider = "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"
        lines = [
            f"### {title}",
            "| " + " | ".join(headers) + " |",
            divider,
        ]
        for row in rows:
            lines.append("| " + " | ".join(row[h] for h in headers) + " |")
        return "\n".join(lines)

    return md_table(ps_rows, SECTION_PS) + "\n\n" + md_table(pb_rows, SECTION_PB)


def write_csv(path: Path, ps_rows: List[Dict[str, str]], pb_rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Scenario"] + TABLE_HEADERS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ps_rows:
            writer.writerow({"Scenario": "PS", **row})
        for row in pb_rows:
            writer.writerow({"Scenario": "PB", **row})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Table 3-style p-hat-driven mixed results for PS and PB (paper model order)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data/synthetic/phat_driven_mixed",
        help="Root p-hat-driven mixed directory (one sub-folder per model).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model keys or paper names. Use 'all' for all 8 models in paper order.",
    )
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--relpe-threshold", type=float, default=0.25,
                        help="RelPE recovery threshold (paper uses 0.25).")
    parser.add_argument("--relpe-eps", type=float, default=DEFAULT_RELP_EPS)
    parser.add_argument(
        "--relpe-mode",
        default=DEFAULT_MAIN_RELPE_MODE,
        choices=["mean_coord", "l1_ratio", "l2_ratio"],
        help="RelPE scalarization. Table 3 uses l1_ratio.",
    )
    parser.add_argument("--cap-lag", type=int, default=DEFAULT_CAP_LAG)
    parser.add_argument(
        "--params-by-model-json",
        default=None,
        help="JSON string or path to hyperparameter file. "
             "Default: configs/model_hyperparameters_synthetic.json (group: evaluation).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/main",
        help="Directory for default output files.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-txt", type=Path, default=None)
    parser.add_argument("--no-output-files", action="store_true",
                        help="Only print to stdout; do not write files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = select_models(args.models)
    params_by_model = (
        load_params_by_model(args.params_by_model_json)
        if args.params_by_model_json
        else DEFAULT_PARAMS_BY_MODEL
    )

    ps_rows: List[Dict[str, str]] = []
    pb_rows: List[Dict[str, str]] = []

    for model_key in models:
        if model_key not in params_by_model:
            raise SystemExit(f"Missing hyperparameters for model '{model_key}'.")

        root = model_root(args.data_dir, model_key)
        scenario_results: Dict[str, Dict[str, Any]] = {}
        for scenario in ("PS", "PB"):
            files = scenario_files(root, scenario)
            scenario_results[scenario] = evaluate_scenario(
                model_key=model_key,
                files=files,
                params=params_by_model[model_key],
                warmup_steps=args.warmup_steps,
                relpe_threshold=args.relpe_threshold,
                relpe_eps=args.relpe_eps,
                relpe_mode=args.relpe_mode,
                cap_lag=args.cap_lag,
            )

        ps_fmt = format_result(scenario_results["PS"], args.relpe_threshold, args.cap_lag)
        pb_fmt = format_result(scenario_results["PB"], args.relpe_threshold, args.cap_lag)
        display = MODEL_DISPLAY[model_key]
        ps_rows.append({"Model": display, **ps_fmt})
        pb_rows.append({"Model": display, **pb_fmt})

    header_lines = [
        f"Table 3 — p-hat-driven mixed selection",
        f"Data:     {args.data_dir}",
        f"RelPE:    threshold={args.relpe_threshold:g}  mode={args.relpe_mode}  eps={args.relpe_eps}",
        f"Hyperparameters: {'custom' if args.params_by_model_json else 'configs/model_hyperparameters_synthetic.json'}",
        "",
    ]

    output_sections = (
        "\n".join(header_lines)
        + "\n"
        + section_table(ps_rows, SECTION_PS)
        + "\n\n"
        + section_table(pb_rows, SECTION_PB)
        + "\n"
        + SECTION_DIVIDER
    )

    print(output_sections)

    if not args.no_output_files:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = args.output_csv or args.output_dir / f"synthetic_main_table3_{stamp}.csv"
        output_txt = args.output_txt or args.output_dir / f"synthetic_main_table3_{stamp}.txt"
        md_text = "\n".join(header_lines) + "\n" + markdown_two_sections(ps_rows, pb_rows)
        write_csv(output_csv, ps_rows, pb_rows)
        write_text(output_txt, md_text)
        print(f"Wrote text : {output_txt}")
        print(f"Wrote CSV  : {output_csv}")


if __name__ == "__main__":
    main()
