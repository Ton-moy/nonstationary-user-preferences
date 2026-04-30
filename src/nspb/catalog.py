"""Item catalog generation and size helpers for the p-hat-driven synthetic setting.

Three item families (K=10 default):
  head_single  — 1 dominant topic:  K  * 50 =   500 items
  head_combo   — 2 dominant topics: C(K,2) * 50 = 2,250 items
  mid_blend    — 4 active topics:   C(K,4) *  1 =   210 items
  Total: 2,960 items

Output layout per family (tab-separated, header on first line):
    item_id \\t alpha \\t topic_vector

Ported from the original notebook/script with parameters aligned to synthetic.yaml.
"""

from __future__ import annotations

import itertools
import uuid
from math import comb
from pathlib import Path
from typing import Optional

import numpy as np


def expected_catalog_size(k: int = 10, samples_per_single: int = 50, samples_per_pair: int = 50) -> int:
    """Return the paper's p-hat-driven catalog size.

    The catalog contains 1-topic, 2-topic, and 4-topic Dirichlet item families:
    K * 50 + C(K, 2) * 50 + C(K, 4) * 1.
    For K=10 this is 2,960.
    """

    return k * samples_per_single + comb(k, 2) * samples_per_pair + comb(k, 4)


def _dirichlet(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sample = rng.gamma(shape=alpha, scale=1.0)
    total = sample.sum()
    if total <= 0:
        return np.ones_like(sample) / len(sample)
    return sample / total


def _alpha_str(alpha: np.ndarray) -> str:
    parts = []
    for a in alpha.tolist():
        if abs(a - round(a)) < 1e-9:
            parts.append(str(int(round(a))))
        else:
            s = f"{float(a):.3f}".rstrip("0").rstrip(".")
            parts.append(s if s else "0")
    return "[" + ",".join(parts) + "]"


def _theta_str(theta: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in theta.tolist()) + "]"


def _write_items(path: Path, alpha: np.ndarray, n: int, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha_s = _alpha_str(alpha)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("item_id\talpha\ttopic_vector\n")
        for _ in range(n):
            theta = _dirichlet(alpha, rng)
            fh.write(f"{uuid.uuid4()}\t{alpha_s}\t{_theta_str(theta)}\n")


def _gen_head_single(
    out_dir: Path,
    k: int,
    samples: int,
    alpha_high: float,
    alpha_other: float,
    rng: np.random.Generator,
) -> None:
    base = out_dir / "head_single"
    for topic in range(1, k + 1):
        alpha = np.full(k, alpha_other)
        alpha[topic - 1] = alpha_high
        _write_items(base / str(topic) / "items.txt", alpha, samples, rng)


def _gen_head_combo(
    out_dir: Path,
    k: int,
    samples: int,
    alpha_high: float,
    alpha_other: float,
    rng: np.random.Generator,
) -> None:
    base = out_dir / "head_combo"
    for i, j in itertools.combinations(range(1, k + 1), 2):
        alpha = np.full(k, alpha_other)
        alpha[i - 1] = alpha_high
        alpha[j - 1] = alpha_high
        _write_items(base / f"{i}_{j}" / "items.txt", alpha, samples, rng)


def _gen_mid_blend(
    out_dir: Path,
    k: int,
    alpha_high: float,
    alpha_other: float,
    rng: np.random.Generator,
) -> None:
    """One item per 4-topic combination (C(K,4) = 210 for K=10)."""
    base = out_dir / "mid_blend"
    for combo in itertools.combinations(range(1, k + 1), 4):
        i, j, kk, l = combo
        alpha = np.full(k, alpha_other)
        for idx in combo:
            alpha[idx - 1] = alpha_high
        _write_items(base / f"{i}_{j}_{kk}_{l}" / "items.txt", alpha, 1, rng)


def generate_catalog(
    output_dir: Path,
    k: int = 10,
    seed: int = 42,
    samples_per_single: int = 50,
    samples_per_pair: int = 50,
    single_alpha_high: float = 20.0,
    single_alpha_other: float = 2.0,
    combo_alpha_high: float = 20.0,
    combo_alpha_other: float = 2.0,
    blend_alpha_high: float = 8.0,
    blend_alpha_other: float = 2.0,
) -> int:
    """Generate the full item catalog under output_dir and return the item count."""

    rng = np.random.default_rng(seed)
    _gen_head_single(output_dir, k, samples_per_single, single_alpha_high, single_alpha_other, rng)
    _gen_head_combo(output_dir, k, samples_per_pair, combo_alpha_high, combo_alpha_other, rng)
    _gen_mid_blend(output_dir, k, blend_alpha_high, blend_alpha_other, rng)
    return expected_catalog_size(k, samples_per_single, samples_per_pair)
