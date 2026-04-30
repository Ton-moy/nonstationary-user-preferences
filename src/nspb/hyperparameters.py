"""Load model hyperparameters from repository JSON config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPERPARAMETER_FILE_SYNTHETIC = REPO_ROOT / "configs" / "model_hyperparameters_synthetic.json"
DEFAULT_HYPERPARAMETER_FILE_REAL = REPO_ROOT / "configs" / "model_hyperparameters_real.json"
DEFAULT_HYPERPARAMETER_FILE = DEFAULT_HYPERPARAMETER_FILE_SYNTHETIC


def _validate_params(obj: Any, source: str) -> Dict[str, Dict[str, float]]:
    if not isinstance(obj, dict):
        raise ValueError(f"Hyperparameters in {source} must be a JSON object.")

    params: Dict[str, Dict[str, float]] = {}
    for model_key, values in obj.items():
        if not isinstance(values, dict):
            raise ValueError(f"Hyperparameters for model '{model_key}' in {source} must be an object.")
        params[str(model_key)] = {str(k): v for k, v in values.items()}
    return params


def _select_group(obj: Any, group: str, source: str) -> Any:
    if isinstance(obj, dict) and group in obj:
        return obj[group]
    return obj


def load_hyperparameter_group(
    group: str,
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, float]]:
    """Load one hyperparameter group from a JSON file.

    If the JSON has a top-level key matching ``group``, that group is used.
    Otherwise the entire JSON object is interpreted as a direct model->params map.
    This keeps old override files compatible.
    """

    json_path = Path(path) if path is not None else DEFAULT_HYPERPARAMETER_FILE
    with json_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    return _validate_params(_select_group(obj, group, str(json_path)), str(json_path))


def load_hyperparameter_arg(
    arg: Optional[str],
    group: str,
) -> Dict[str, Dict[str, float]]:
    """Load hyperparameters from a path, JSON string, or default config."""

    if arg is None:
        return load_hyperparameter_group(group)

    candidate = Path(arg)
    if candidate.is_file():
        return load_hyperparameter_group(group, candidate)

    try:
        obj = json.loads(arg)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected a JSON string or file path for hyperparameters: {arg[:120]}") from exc

    return _validate_params(_select_group(obj, group, "JSON argument"), "JSON argument")
