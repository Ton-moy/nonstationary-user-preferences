#!/usr/bin/env python3
"""
Generate p-hat-driven user interaction logs.

Behavior
--------
• Preference-change steps (11, 21, 31, ...): keep the provided topic_vector.
• Stable steps: select generated_topic_vector = argmax_{theta in catalog} p_t^T theta,
  (Using Ground Truth preference vector p_t as per user instruction).
• Outcome rating uses the ground-truth preference vector: y_t = p_t^T theta_t.
• Update the model with (theta_t, y_t) after each step.
• Output JSON per user record contains ONLY:
    - "generated_topic_vector"
    - "rating"
    - "preference_vector"             (ground truth)
    - "preference_change_label"       (1 at 11/21/31/..., else 0)
• If source logs omit "timestep", records are treated as timesteps 1..T.

Catalog format
--------------
Recursively under --catalog-dir, each items.txt has header:
    item_id \t alpha \t topic_vector
We parse only the 3rd column (topic_vector) into a K-dim row. All rows must share the same K.
"""

from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------
# Project schema keys
# ----------------------------
PREFERENCE_CHANGE_LABEL_KEY = "preference_change_label"
SOURCE_X_KEY = "topic_vector"   # present at preference-change steps in source data
GENERATED_X_KEY = "generated_topic_vector"
GT_KEY = "preference_vector"    # provided for all steps
T_KEY = "timestep"

# ----------------------------
# Import your models
# ----------------------------
from nspb.models import (
    BayesianModel,
    VarianceBoundedBayesianModel,
    BayesianForgettingFactorModel,
    AROW_Regression,
    KalmanFilter,
    NLMS,  # not used here
    BayesianSlidingWindowModel,
    PowerPriorBayesianModel,
    NormalWishartBayesianModel,
)
from nspb.hyperparameters import load_hyperparameter_arg, load_hyperparameter_group

# ----------------------------
# Tuned hyperparameters per model
# ----------------------------
PARAMS_BY_MODEL: Dict[str, Dict[str, float]] = load_hyperparameter_group("phat_generation")

# ----------------------------
# Preference-change schedule (fixed-cycle)
# ----------------------------
def is_preference_change_step(t: int) -> bool:
    # 11, 21, 31, ... (t starts at 1)
    return t >= 11 and ((t - 1) % 10 == 0)

# ----------------------------
# Catalog loading
# ----------------------------
def _parse_bracket_vector(text: str) -> np.ndarray:
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"Expected bracketed vector, got: {text[:80]}...")
    inner = s[1:-1].strip()
    if not inner:
        return np.array([], dtype=float)
    parts = inner.split(",")
    return np.asarray([float(p) for p in parts], dtype=float)

def load_catalog_topic_vectors(catalog_root: Path) -> np.ndarray:
    """
    Recursively load all items.txt and return a stacked matrix of topic vectors: (M, K).
    Assumes header line: 'item_id\\talpha\\ttopic_vector'.
    """
    files = [Path(p) for p in glob.glob(str(catalog_root / "**/items.txt"), recursive=True)]
    if not files:
        raise FileNotFoundError(f"No items.txt found under: {catalog_root}")
    all_vecs: List[np.ndarray] = []
    K: Optional[int] = None
    for fp in files:
        with fp.open("r", encoding="utf-8") as fh:
            _ = fh.readline()  # header
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                theta = _parse_bracket_vector(parts[2])
                if theta.size == 0:
                    continue
                if K is None:
                    K = theta.size
                elif theta.size != K:
                    raise ValueError(f"Mixed dimensionality in catalog: saw {theta.size}, expected {K}")
                all_vecs.append(theta)
    if not all_vecs:
        raise ValueError("Catalog empty or malformed (no topic vectors parsed)")
    return np.vstack(all_vecs)  # (M, K)

# ----------------------------
# Model factory & current-mean accessor
# ----------------------------
def build_model(model_key: str, K: int, params: Dict[str, float], initial_mean: Optional[np.ndarray] = None):
    mk = model_key
    model = None
    
    if mk == "KF":
        model = KalmanFilter(K,
                             params.get("variance_p", 3.0),
                             params.get("variance", 0.3),
                             params.get("delta", 0.015),
                             params.get("eta", 1e-5))
    elif mk == "BLR":
        model = BayesianModel(K,
                              params.get("default_variance", 50.0),
                              params.get("noise_precision", 0.33))
    elif mk == "vbBLR":
        tau_param = params.get("tau_v", params.get("tau", 1.0))
        model = VarianceBoundedBayesianModel(K,
                                             params.get("default_variance", 15.0),
                                             params.get("noise_precision", 0.33),
                                             tau_param)
    elif mk == "fBLR":
        model = BayesianForgettingFactorModel(K,
                                              params.get("default_variance", 50.0),
                                              params.get("noise_precision", 0.33))
    elif mk == "AROW":
        model = AROW_Regression(K,
                                params.get("lam1", 3.0),
                                params.get("lam2", 4.0))
    elif mk == "BLRsw":
        model = BayesianSlidingWindowModel(K,
                                           params.get("m", 20),
                                           params.get("default_variance", 50.0),
                                           params.get("noise_precision", 0.33))
    elif mk == "PBLR":
        model = PowerPriorBayesianModel(K,
                                        params.get("alpha", 0.5),
                                        params.get("default_variance", 50.0),
                                        params.get("noise_precision", 0.33))
    elif mk == "NIW":
        # NIW supports m0 in constructor
        m0_val = initial_mean if initial_mean is not None else None
        model = NormalWishartBayesianModel(K,
                                           m0=m0_val,
                                           kappa0=params.get("kappa0", 1.0),
                                           nu0=params.get("nu0", 3.0),
                                           S0=None)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    # For models other than NIW, manually set the mean if provided and attribute exists
    if initial_mean is not None and mk != "NIW":
        # Specific handling for BLRsw which uses 'p' and '_initial_p'
        # (and implies 'm' is window size, so we must NOT overwrite 'm')
        if mk == "BLRsw":
            if hasattr(model, 'p'):
                setattr(model, 'p', initial_mean.copy())
            if hasattr(model, '_initial_p'):
                setattr(model, '_initial_p', initial_mean.copy())
        
        # Common attribute 'mu' for BLR, vbBLR, PBLR, fBLR
        elif hasattr(model, 'mu'):
            setattr(model, 'mu', initial_mean.copy())
            
        # Fallback for others (e.g. KF might use 'm' for state mean)
        # We ensure we don't overwrite 'm' if it's actually an integer config param
        elif hasattr(model, 'm'):
            # Only overwrite if it looks like a vector (or model is known to use m as mean)
            # Safe guard: don't overwrite if existing m is int/float
            existing_m = getattr(model, 'm')
            if not isinstance(existing_m, (int, float)):
                setattr(model, 'm', initial_mean.copy())
            
    return model

def current_mean_vector(model, model_key: str) -> np.ndarray:
    if model_key == "NIW":
        # get_params() -> (k, m, v, S)
        _, m, _, _ = model.get_params()
        return np.asarray(m, dtype=float).ravel()
    else:
        mu, *_ = model.get_params()
        return np.asarray(mu, dtype=float).ravel()

# ----------------------------
# Selection (deterministic argmax) + print 2nd highest
# ----------------------------
def select_argmax_hatp_dot_theta(hat_p: np.ndarray, catalog: np.ndarray) -> Tuple[np.ndarray, int, float]:
    scores = catalog @ hat_p  # (M,)
    M = scores.shape[0]
    if M == 0:
        raise ValueError("Catalog is empty.")
    if M == 1:
        idx = 0
        top_score = float(scores[idx])
        return catalog[idx], idx, top_score

    # Top-2 indices efficiently (unordered), then sort by score desc
    top2 = np.argpartition(scores, -2)[-2:]
    top2 = top2[np.argsort(scores[top2])[::-1]]  # highest first
    idx, second_idx = int(top2[0]), int(top2[1])
    top_score = float(scores[idx])
    second_score = float(scores[second_idx])

    return catalog[idx], idx, top_score

# ----------------------------
# I/O helpers for user files
# ----------------------------
def read_user_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_user_json(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

def record_timestep(record: dict, index: int) -> int:
    value = record.get(T_KEY)
    if value is None:
        return index + 1
    return int(value)

def collect_user_files(root: Path) -> List[Path]:
    paths: List[Path] = []
    paths.extend(Path(root).glob("*.json"))       # top-level files
    paths.extend(Path(root).glob("*/*.json"))     # one-level subfolders
    return sorted(paths)

# ----------------------------
# Per-user processing
# ----------------------------
def process_user_file(src_path: Path, dst_path: Path, catalog: np.ndarray,
                      model_key: str, params: Dict[str, float]) -> None:
    data = read_user_json(src_path)
    if not data:
        write_user_json(dst_path, [])
        return

    # 1. Look for timestep 0 to set initial preference
    initial_preference = None
    for idx, rec in enumerate(data):
        if record_timestep(rec, idx) == 0 and GT_KEY in rec:
            initial_preference = np.asarray(rec[GT_KEY], dtype=float)
            break
    if initial_preference is None and GT_KEY in data[0]:
        initial_preference = np.asarray(data[0][GT_KEY], dtype=float)
            
    # Validate K using the first non-zero timestep or the initial pref
    sample_rec = next((r for idx, r in enumerate(data) if record_timestep(r, idx) > 0), None)
    if sample_rec:
        K = len(sample_rec[GT_KEY])
    elif initial_preference is not None:
        K = len(initial_preference)
    else:
        # Fallback if file has no valid data
        K = catalog.shape[1]

    if catalog.shape[1] != K:
        raise ValueError(f"Catalog K={catalog.shape[1]} != preference K={K} in {src_path}")

    # Build map of given topics at cycle preference-change steps (11,21,31,...)
    given_topics: Dict[int, np.ndarray] = {}
    for idx, rec in enumerate(data):
        t = record_timestep(rec, idx)
        # Skip timestep 0 for this map
        if t <= 0:
            continue
        if is_preference_change_step(t):
            if SOURCE_X_KEY not in rec or rec[SOURCE_X_KEY] is None:
                raise KeyError(f"Preference-change step t={t} (by rule) missing topic_vector in {src_path}")
            given_topics[t] = np.asarray(rec[SOURCE_X_KEY], dtype=float)

    # Initialize model with the initial preference if found
    model = build_model(model_key, K, params, initial_mean=initial_preference)

    out_records: List[dict] = []

    # Process in ascending timestep order
    # We sort strictly to ensure order
    sorted_data = sorted(
        ((record_timestep(rec, idx), idx, rec) for idx, rec in enumerate(data)),
        key=lambda item: (item[0], item[1]),
    )
    
    for t, _, rec in sorted_data:
        if t == 0:
            continue
            
        p_t = np.asarray(rec[GT_KEY], dtype=float) # Ground Truth Preference
        is_preference_change = is_preference_change_step(t)

        if is_preference_change:
            theta_t = given_topics[t]
            preference_change_label = 1
        else:
            # Select item using Ground Truth (p_t)
            theta_t, _, _ = select_argmax_hatp_dot_theta(p_t, catalog)
            preference_change_label = 0

        # Outcome uses Ground Truth: y_t = p_t^T * theta_t
        y_t = float(np.dot(p_t, theta_t))
        
        if model_key == "fBLR":
            rho = float(params.get("rho", 0.98))
            model.update(theta_t, y_t, rho=rho)
        else:
            model.update(theta_t, y_t)

        # Output the compact public schema used by evaluation/tuning scripts.
        out_rec = {
            GENERATED_X_KEY: theta_t.tolist(),
            "rating": y_t,
            GT_KEY: rec[GT_KEY],
            PREFERENCE_CHANGE_LABEL_KEY: preference_change_label,
        }
        out_records.append(out_rec)

    write_user_json(dst_path, out_records)

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate p-hat logs by model-based selection with cycle preference-change steps (11,21,31,...).")
    ap.add_argument("--input-dir", required=True, type=Path, help="Dataset root with per-user JSON files (has subfolders)")
    ap.add_argument("--output-dir", required=True, type=Path, help="Output root; subfolders mirrored from input")
    ap.add_argument("--catalog-dir", required=True, type=Path, help="Root containing head_single/head_combo/mid_blend items.txt")
    ap.add_argument("--model", required=True,
                    choices=["KF", "BLR", "vbBLR", "fBLR", "AROW", "BLRsw", "PBLR", "NIW"],
                    help="Model key to use for selection")
    ap.add_argument("--params-by-model-json", default=None, help="JSON string or JSON file path for model hyperparameters.")
    args = ap.parse_args()

    # Tuned hyperparameters for the chosen model
    params_by_model = load_hyperparameter_arg(args.params_by_model_json, "phat_generation")
    if args.model not in params_by_model:
        raise ValueError(f"No tuned hyperparameters registered for model {args.model}")
    params: Dict[str, float] = dict(params_by_model[args.model])

    # Load catalog
    catalog = load_catalog_topic_vectors(args.catalog_dir)
    print(f"Catalog loaded: {catalog.shape[0]} items, K={catalog.shape[1]}")

    # Traverse input → output (mirrored structure)
    src_files = collect_user_files(args.input_dir)
    if not src_files:
        print(f"No JSON files found under {args.input_dir}")
        return

    for src in src_files:
        rel = src.relative_to(args.input_dir)
        dst = args.output_dir / rel
        process_user_file(src, dst, catalog, args.model, params)
        print(f"Wrote: {dst}")

    print("Done.")

if __name__ == "__main__":
    main()
